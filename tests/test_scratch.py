"""Tests for the from-scratch byte-level BiLSTM model and its training script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from newlinefix.data import write_documents
from newlinefix.gaps import NUM_GAP_CLASSES, SPACE, text_to_gaps
from newlinefix.models.scratch import (
    MAX_BYTES,
    CharBiLSTM,
    ScratchGapPredictor,
    encode_window,
    gap_byte_positions,
)
from newlinefix.predict import fix_text

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_scratch.py"


def tiny_predictor(seed: int = 0) -> ScratchGapPredictor:
    torch.manual_seed(seed)
    model = CharBiLSTM(embed_size=8, hidden_size=16, num_layers=1, dropout=0.0)
    return ScratchGapPredictor(model, device="cpu")


def test_gap_byte_positions_ascii() -> None:
    words = ["hello", "big", "world"]
    raw = " ".join(words).encode("utf-8")
    positions = gap_byte_positions(words)
    assert positions == [5, 9]
    assert all(raw[p] == 0x20 for p in positions)


def test_gap_byte_positions_multibyte() -> None:
    words = ["•", "café", "naïve", "é", "ok"]
    raw = " ".join(words).encode("utf-8")
    positions = gap_byte_positions(words)
    assert len(positions) == len(words) - 1
    assert all(raw[p] == 0x20 for p in positions)
    # encode_window shifts every byte by +1, so a space reads as 0x21 there.
    ids = encode_window(words)
    assert all(ids[p] == 0x20 + 1 for p in positions)


def test_gap_byte_positions_degenerate() -> None:
    assert gap_byte_positions([]) == []
    assert gap_byte_positions(["only"]) == []


def test_untrained_predict_window_shape_and_range() -> None:
    predictor = tiny_predictor()
    words = [f"word{i}" for i in range(30)] + ["•", "café"]
    labels = predictor.predict_window(words)
    assert len(labels) == len(words) - 1
    assert all(0 <= lab < NUM_GAP_CLASSES for lab in labels)
    assert predictor.predict_window(["single"]) == []


def test_predict_window_beyond_byte_cap_falls_back_to_space() -> None:
    predictor = tiny_predictor()
    words = ["a" * 30 for _ in range(150)]  # ~4.6 KB, well past MAX_BYTES
    positions = gap_byte_positions(words)
    labels = predictor.predict_window(words)
    assert len(labels) == len(words) - 1
    over_cap = [lab for pos, lab in zip(positions, labels, strict=True) if pos >= MAX_BYTES]
    assert over_cap
    assert all(lab == SPACE for lab in over_cap)


def test_predict_windows_batched() -> None:
    predictor = tiny_predictor()
    w1 = ["alpha", "beta", "gamma", "delta"]
    w2 = [f"tok{i}" for i in range(12)]
    w3 = ["x"]
    results = predictor.predict_windows([w1, w2, w3])
    assert [len(r) for r in results] == [3, 11, 0]
    assert all(0 <= lab < NUM_GAP_CLASSES for r in results for lab in r)
    assert predictor.predict_windows([]) == []
    assert predictor.predict_windows([w2])[0] == predictor.predict_window(w2)


def test_save_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = CharBiLSTM(embed_size=8, hidden_size=16, num_layers=2, dropout=0.2)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    config = {
        "embed_size": 8,
        "hidden_size": 16,
        "num_layers": 2,
        "dropout": 0.2,
        "max_words": 120,
        "overlap": 40,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    original = ScratchGapPredictor(model, device="cpu")
    loaded = ScratchGapPredictor.load(tmp_path, device="cpu")
    assert loaded.max_words == 120
    assert loaded.overlap == 40
    words = ["The", "quick", "•", "brown", "café", "jumps"] * 5
    assert loaded.predict_window(words) == original.predict_window(words)


def _synthetic_docs() -> list[dict[str, str]]:
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "café", "zeta", "eta", "theta"]
    docs: list[dict[str, str]] = []
    for d in range(24):
        paragraphs: list[str] = []
        for p in range(6):  # 6 paragraphs x 5 lines x 10 words = 300 words per doc
            lines: list[str] = []
            for li in range(5):
                start = (d + 3 * p + li) % len(vocab)
                line_words = [vocab[(start + k) % len(vocab)] for k in range(10)]
                if li == 0:
                    line_words[0] = "•"
                lines.append(" ".join(line_words))
            paragraphs.append("\n".join(lines))
        docs.append({"text": "\n\n".join(paragraphs), "source": "synthetic"})
    return docs


def test_train_smoke_and_fix_text(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs = _synthetic_docs()
    write_documents(docs_dir / "train.jsonl", docs[:20])
    write_documents(docs_dir / "val.jsonl", docs[20:])
    out_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        *("--data", str(docs_dir)),
        *("--out", str(out_dir)),
        *("--embed-size", "8", "--hidden-size", "16", "--layers", "1", "--dropout", "0.0"),
        *("--epochs", "1", "--batch-size", "8"),
        *("--train-windows", "32", "--val-windows", "8"),
        *("--device", "cpu", "--log-every", "2"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "model.pt").exists()
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["max_words"] == 150
    assert config["overlap"] == 50
    assert "val_metrics" in config
    assert "accuracy" in config["val_metrics"]
    log = json.loads((out_dir / "training_log.json").read_text(encoding="utf-8"))
    assert log["steps"] and {"step", "loss", "lr"} <= log["steps"][0].keys()
    assert log["epochs"] and "macro_f1_structural" in log["epochs"][0]

    predictor = ScratchGapPredictor.load(out_dir, device="cpu")
    words = [f"w{i}" for i in range(500)]  # 500 words > max_words -> exercises windowing
    broken = (
        "".join(w + ("\n" if i % 17 == 3 else " ") for i, w in enumerate(words[:-1])) + words[-1]
    )
    fixed = fix_text(broken, predictor)
    assert text_to_gaps(fixed).words == words
