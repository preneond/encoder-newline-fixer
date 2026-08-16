import json
import random
from pathlib import Path

from newlinefix.corruption import CorruptionConfig
from newlinefix.data import doc_to_windows, load_training_windows, read_documents, write_documents

DOC = (
    "Heading One\n\nSome first paragraph with several words in it.\n• a bullet item\n• another one"
)


def test_write_and_read_documents_round_trip(tmp_path: Path):
    path = tmp_path / "docs" / "train.jsonl"
    n = write_documents(path, [{"text": DOC, "source": "test"}, {"text": "a b", "source": "test"}])
    assert n == 2
    assert list(read_documents(path)) == [DOC, "a b"]
    # ensure_ascii=False keeps the bullet readable on disk
    assert "•" in path.read_text(encoding="utf-8")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_doc_to_windows_alignment():
    rng = random.Random(0)
    cfg = CorruptionConfig(p_word_split=0.3)
    windows = list(doc_to_windows(DOC, max_words=5, rng=rng, cfg=cfg))
    assert windows
    for words, labels in windows:
        assert 2 <= len(words) <= 5
        assert len(labels) == len(words) - 1
    # Window words concatenate back to the full corrupted word sequence.
    all_words = [w for words, _ in windows for w in words]
    rng2 = random.Random(0)
    full = list(doc_to_windows(DOC, max_words=10_000, rng=rng2, cfg=cfg))
    assert len(full) == 1
    assert all_words == full[0][0]


def test_load_training_windows_deterministic_and_limited(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    docs = [{"text": " ".join(f"word{i}" for i in range(50)), "source": "t"} for _ in range(20)]
    with open(path, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    a = load_training_windows(path, max_words=8, seed=7)
    b = load_training_windows(path, max_words=8, seed=7)
    assert a == b
    assert len(load_training_windows(path, max_words=8, seed=7, limit=5)) == 5
    # different seed -> different corruption/order
    c = load_training_windows(path, max_words=8, seed=8)
    assert a != c
