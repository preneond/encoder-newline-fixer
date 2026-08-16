"""Knowledge distillation from a fine-tuned encoder teacher into small students.

The teacher is a trained token-classification encoder (artifacts written by
scripts/train_encoder.py). For every training window it emits per-gap soft
targets aligned with the student's gap order; the student adds a masked,
temperature-scaled KL term to its usual hard-label cross-entropy.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from newlinefix.data import Window

IGNORE_INDEX = -100


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    valid: Tensor,
    labels: Tensor,
    temperature: float,
) -> Tensor:
    """Masked soft-target loss KL(teacher || student), scaled by temperature^2.

    All tensors are gap-aligned: logits are [B, G, C]; ``valid`` [B, G] marks
    gaps the teacher actually scored; ``labels`` [B, G] carries IGNORE_INDEX at
    padded gaps. Gaps failing either mask contribute nothing. The temperature^2
    factor keeps soft-target gradients on the same scale as the hard CE term
    (Hinton et al., 2015).
    """
    tau = temperature
    kl = nn.functional.kl_div(
        torch.log_softmax(student_logits / tau, dim=-1),
        torch.softmax(teacher_logits / tau, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    mask = (valid & (labels != IGNORE_INDEX)).to(kl.dtype)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0) * (tau * tau)


class EncoderTeacher:
    """Frozen fine-tuned encoder that scores every gap of a training batch."""

    def __init__(self, artifact_dir: Path | str, device: torch.device) -> None:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            PreTrainedTokenizerBase,
        )

        self.tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(artifact_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(artifact_dir)
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def gap_logits(self, batch: list[Window], n_gap_cols: int) -> tuple[Tensor, Tensor]:
        """Teacher logits [B, G, C] and validity mask [B, G]; gap i sits at column i.

        Column layout matches the student collate: gaps come in document order,
        so a student that keeps only a prefix of gaps (byte-cap truncation) still
        lines up. A gap is invalid when the teacher's 512-token truncation cut
        off the last subtoken of its left word.
        """
        from newlinefix.models.encoder import last_subtoken_positions

        enc = self.tokenizer(
            [words for words, _ in batch],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        logits = self.model(**enc.to(self.device)).logits
        out = torch.zeros(len(batch), n_gap_cols, logits.size(-1), device=logits.device)
        valid = torch.zeros(len(batch), n_gap_cols, dtype=torch.bool, device=logits.device)
        for b, (words, _) in enumerate(batch):
            positions = last_subtoken_positions(enc.word_ids(batch_index=b), len(words))
            for i in range(min(len(words) - 1, n_gap_cols)):
                pos = positions[i]
                if pos is not None:
                    out[b, i] = logits[b, pos]
                    valid[b, i] = True
        return out, valid
