"""Fine-tuned pretrained encoder for token-gap classification.

Gap i (between words[i] and words[i+1]) is classified from the token-classification
logits at the *last subtoken of word i* (the left word). Words whose subtokens were
truncated away by the 512-token limit fall back to SPACE for their gaps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from newlinefix.gaps import SPACE
from newlinefix.predict import GapPredictor


def pick_device(device: str | None = None) -> str:
    """Explicit device, else auto-select mps > cuda > cpu."""
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def last_subtoken_positions(word_ids: list[int | None], n_words: int) -> list[int | None]:
    """For each word index, the sequence position of its last subtoken.

    ``word_ids`` is ``BatchEncoding.word_ids`` output (None for special/padding
    tokens). A word truncated away entirely maps to None.
    """
    positions: list[int | None] = [None] * n_words
    for pos, wid in enumerate(word_ids):
        if wid is not None and 0 <= wid < n_words:
            positions[wid] = pos
    return positions


class EncoderGapPredictor(GapPredictor):
    """Serves a fine-tuned AutoModelForTokenClassification checkpoint."""

    max_words = 180
    overlap = 64

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, device: str
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> EncoderGapPredictor:
        dev = pick_device(device)
        # add_prefix_space is required by BPE tokenizers (RoBERTa family) for
        # pretokenized input; other tokenizers store-and-ignore the kwarg.
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(str(path), add_prefix_space=True),
        )
        model = AutoModelForTokenClassification.from_pretrained(str(path))
        model.to(dev)
        model.eval()
        predictor = cls(tokenizer, model, dev)
        # Honor the windowing the checkpoint was trained with (mirrors ScratchGapPredictor).
        config_path = Path(path) / "predictor_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            predictor.max_words = int(config.get("max_words", cls.max_words))
            predictor.overlap = int(config.get("overlap", cls.overlap))
        return predictor

    def predict_window(self, words: list[str]) -> list[int]:
        return self.predict_windows([words])[0]

    def predict_windows(self, windows: list[list[str]]) -> list[list[int]]:
        """Batched prediction: one list of len(words)-1 gap labels per window."""
        if not windows:
            return []
        enc = self.tokenizer(
            windows,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self.model(**enc.to(self.device)).logits
        preds = logits.argmax(dim=-1).cpu()
        out: list[list[int]] = []
        for b, words in enumerate(windows):
            positions = last_subtoken_positions(enc.word_ids(batch_index=b), len(words))
            labels: list[int] = []
            for i in range(len(words) - 1):
                pos = positions[i]
                labels.append(SPACE if pos is None else int(preds[b, pos]))
            out.append(labels)
        return out
