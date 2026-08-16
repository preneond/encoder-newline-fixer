"""Offline tests for the distillation loss (no models required)."""

from __future__ import annotations

import pytest
import torch

from newlinefix.distill import IGNORE_INDEX, distillation_loss


def _uniform(b: int, g: int, c: int = 4) -> torch.Tensor:
    return torch.zeros(b, g, c)


def test_matching_distributions_give_zero_loss() -> None:
    logits = torch.randn(2, 5, 4)
    valid = torch.ones(2, 5, dtype=torch.bool)
    labels = torch.zeros(2, 5, dtype=torch.long)
    loss = distillation_loss(logits, logits.clone(), valid, labels, temperature=2.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_mismatched_distributions_give_positive_loss() -> None:
    student = _uniform(1, 3)
    teacher = torch.zeros(1, 3, 4)
    teacher[..., 2] = 5.0  # teacher is confident about class 2
    valid = torch.ones(1, 3, dtype=torch.bool)
    labels = torch.zeros(1, 3, dtype=torch.long)
    loss = distillation_loss(student, teacher, valid, labels, temperature=1.0)
    assert float(loss) > 0.1


def test_invalid_and_padded_gaps_are_masked_out() -> None:
    student = _uniform(1, 4)
    teacher = torch.zeros(1, 4, 4)
    teacher[..., 1] = 10.0
    labels = torch.zeros(1, 4, dtype=torch.long)

    # Only gap 0 counts: gap 1 is teacher-invalid, gaps 2-3 are label-padded.
    valid = torch.tensor([[True, False, True, True]])
    labels[0, 2] = IGNORE_INDEX
    labels[0, 3] = IGNORE_INDEX
    partial = distillation_loss(student, teacher, valid, labels, temperature=1.0)

    only_first = distillation_loss(
        student[:, :1],
        teacher[:, :1],
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.long),
        temperature=1.0,
    )
    assert float(partial) == pytest.approx(float(only_first), rel=1e-6)


def test_fully_masked_batch_is_zero_not_nan() -> None:
    student = torch.randn(2, 3, 4)
    teacher = torch.randn(2, 3, 4)
    valid = torch.zeros(2, 3, dtype=torch.bool)
    labels = torch.zeros(2, 3, dtype=torch.long)
    loss = distillation_loss(student, teacher, valid, labels, temperature=2.0)
    assert float(loss) == 0.0


def test_temperature_scaling_keeps_gradient_magnitude() -> None:
    """The tau^2 factor: loss at high tau stays on a comparable scale, not ~0."""
    student = torch.randn(1, 6, 4)
    teacher = student + torch.randn(1, 6, 4)
    valid = torch.ones(1, 6, dtype=torch.bool)
    labels = torch.zeros(1, 6, dtype=torch.long)
    low = distillation_loss(student, teacher, valid, labels, temperature=1.0)
    high = distillation_loss(student, teacher, valid, labels, temperature=4.0)
    assert float(high) > 0.0
    # Without the tau^2 factor, high-temperature KL collapses ~quadratically.
    assert float(high) > 0.05 * float(low)
