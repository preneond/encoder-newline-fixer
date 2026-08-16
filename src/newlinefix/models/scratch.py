"""From-scratch comparison model: a byte-level BiLSTM gap classifier.

No pretraining — the model reads the raw UTF-8 bytes of a window's space-joined
text and is trained only on our corrupted-window data. The gap between
words[i] and words[i+1] is classified from the BiLSTM output at the byte
position of the space that separates them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from newlinefix.gaps import NUM_GAP_CLASSES, SPACE
from newlinefix.predict import GapPredictor

#: 0 is reserved for padding; byte value b maps to id b + 1.
VOCAB_SIZE = 257
#: Byte-length cap per window; gaps at positions beyond it fall back to SPACE.
MAX_BYTES = 2048


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Map "auto"/None to the best available backend (CUDA > MPS > CPU)."""
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # LSTM runs natively on MPS in torch 2.13.
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def encode_window(words: list[str]) -> list[int]:
    """Byte ids (b + 1) of the space-joined window text."""
    return [b + 1 for b in " ".join(words).encode("utf-8")]


def gap_byte_positions(words: list[str]) -> list[int]:
    """Byte position of the space separating words[i] and words[i+1], for each gap i.

    Positions index into ``encode_window(words)`` (equivalently the UTF-8 bytes of
    the joined text) and always point at a 0x20 byte; word lengths are counted in
    encoded bytes, so multi-byte characters are handled correctly.
    """
    positions: list[int] = []
    pos = 0
    for word in words[:-1]:
        pos += len(word.encode("utf-8"))
        positions.append(pos)
        pos += 1  # step over the separator space itself
    return positions


class CharBiLSTM(nn.Module):
    """Byte embedding -> multi-layer BiLSTM -> 2-layer MLP head over gap classes."""

    def __init__(
        self,
        embed_size: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            # nn.LSTM warns if dropout is set with a single layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_GAP_CLASSES),
        )

    def forward(self, ids: Tensor) -> Tensor:
        """ids [B, L] of byte ids (0 = padding) -> per-byte gap logits [B, L, 4].

        On CPU/CUDA, sequences are packed so the backward LSTM direction never
        reads padding: outputs at real positions are identical regardless of batch
        composition. MPS has no packed-sequence kernel (packing falls back to a
        ~20x-slower path), so it runs the plain padded forward; there, training is
        unaffected (padded positions are masked out of the loss) and serving is
        unaffected (fix_text predicts unpadded single windows) — only *batched*
        multi-length inference on MPS carries a marginal padding influence in the
        backward direction.
        """
        emb = self.embedding(ids)
        if ids.device.type == "mps":
            out, _ = self.lstm(emb)
        else:
            lengths = (ids != 0).sum(dim=1).clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=ids.size(1)
            )
        return self.head(out)

    def config(self) -> dict[str, int | float]:
        """Constructor arguments, as stored in a checkpoint's config.json."""
        return {
            "embed_size": self.embed_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class ScratchGapPredictor(GapPredictor):
    """GapPredictor backed by a trained :class:`CharBiLSTM` checkpoint."""

    max_words = 150
    overlap = 50

    def __init__(self, model: CharBiLSTM, device: str | torch.device | None = None) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def load(
        cls, path: Path | str, device: str | torch.device | None = None
    ) -> ScratchGapPredictor:
        """Load artifacts written by scripts/train_scratch.py (model.pt + config.json)."""
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        model = CharBiLSTM(
            embed_size=int(cfg["embed_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        )
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        predictor = cls(model, device)
        predictor.max_words = int(cfg.get("max_words", cls.max_words))
        predictor.overlap = int(cfg.get("overlap", cls.overlap))
        return predictor

    def predict_window(self, words: list[str]) -> list[int]:
        return self.predict_windows([words])[0]

    def predict_windows(self, windows: list[list[str]]) -> list[list[int]]:
        """Batched prediction with right-padding; one label list per input window."""
        results: list[list[int]] = [[] for _ in windows]
        encoded: list[tuple[int, list[int], list[int]]] = []
        for i, words in enumerate(windows):
            if len(words) < 2:
                continue
            encoded.append((i, encode_window(words)[:MAX_BYTES], gap_byte_positions(words)))
        if not encoded:
            return results
        max_len = max(len(ids) for _, ids, _ in encoded)
        batch = torch.zeros(len(encoded), max_len, dtype=torch.long)
        for row, (_, ids, _) in enumerate(encoded):
            batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        with torch.inference_mode():
            preds = self.model(batch.to(self.device)).argmax(dim=-1).cpu()
        for row, (i, ids, positions) in enumerate(encoded):
            # Gaps truncated away by the MAX_BYTES cap fall back to SPACE.
            results[i] = [int(preds[row, p]) if p < len(ids) else SPACE for p in positions]
        return results
