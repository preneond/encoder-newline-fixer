"""Offline tests for newlinefix.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from newlinefix.gaps import JOIN, NEWLINE, PARA, SPACE, GapText, gaps_to_text
from newlinefix.metrics import (
    accuracy,
    break_prf,
    confusion_matrix,
    edit_similarity,
    macro_f1,
    per_class_prf,
    pk,
    windowdiff,
)

# 7 words, all four gap classes present.
TRUE = [JOIN, SPACE, NEWLINE, SPACE, PARA, SPACE]
WORDS = ["Head", "line", "one", "bullet", "point", "next", "para"]


def test_perfect_prediction() -> None:
    cm = confusion_matrix(TRUE, TRUE)
    assert accuracy(cm) == 1.0
    assert macro_f1(cm, [JOIN, NEWLINE, PARA]) == 1.0
    assert pk(TRUE, TRUE) == 0.0
    assert windowdiff(TRUE, TRUE) == 0.0
    text = gaps_to_text(GapText(WORDS, TRUE))
    assert edit_similarity(text, text) == 1.0
    assert text == gaps_to_text(GapText(WORDS, list(TRUE)))  # exact match


def test_all_space_prediction_has_zero_break_recall() -> None:
    pred = [SPACE] * len(TRUE)
    result = break_prf(TRUE, pred)
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0  # no predicted positives -> defined as 0
    assert result["f1"] == 0.0


def test_break_prf_forgives_newline_para_confusion() -> None:
    pred = [JOIN, SPACE, PARA, SPACE, NEWLINE, SPACE]  # breaks placed right, kinds swapped
    result = break_prf(TRUE, pred)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_confusion_matrix_and_prf_hand_case() -> None:
    true = [0, 0, 1, 1, 2]
    pred = [0, 1, 1, 2, 2]
    cm = confusion_matrix(true, pred)
    expected = np.zeros((4, 4), dtype=np.int64)
    expected[0, 0] = 1
    expected[0, 1] = 1
    expected[1, 1] = 1
    expected[1, 2] = 1
    expected[2, 2] = 1
    assert np.array_equal(cm, expected)
    assert accuracy(cm) == pytest.approx(3 / 5)

    prf = per_class_prf(cm)
    assert prf["JOIN"]["precision"] == pytest.approx(1.0)
    assert prf["JOIN"]["recall"] == pytest.approx(0.5)
    assert prf["JOIN"]["f1"] == pytest.approx(2 / 3)
    assert prf["JOIN"]["support"] == 2
    assert prf["SPACE"]["precision"] == pytest.approx(0.5)
    assert prf["SPACE"]["recall"] == pytest.approx(0.5)
    assert prf["SPACE"]["f1"] == pytest.approx(0.5)
    assert prf["NEWLINE"]["precision"] == pytest.approx(0.5)
    assert prf["NEWLINE"]["recall"] == pytest.approx(1.0)
    assert prf["NEWLINE"]["f1"] == pytest.approx(2 / 3)
    assert prf["PARA"]["support"] == 0
    assert prf["PARA"]["f1"] == 0.0

    assert macro_f1(cm, [0, 1, 2]) == pytest.approx((2 / 3 + 1 / 2 + 2 / 3) / 3)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0])
    with pytest.raises(ValueError):
        break_prf([0, 1], [0])
    with pytest.raises(ValueError):
        pk([0, 1], [0])
    with pytest.raises(ValueError):
        windowdiff([0, 1], [0])


def test_edit_similarity_hand_cases() -> None:
    # One SPACE vs NEWLINE substitution: distance 1, max length 3.
    assert edit_similarity("a b", "a\nb") == pytest.approx(1 - 1 / 3)
    # SPACE vs PARA: substitute + insert = 2, max length 4.
    assert edit_similarity("a b", "a\n\nb") == pytest.approx(1 - 2 / 4)
    # NEWLINE vs PARA: one insertion, max length 4.
    assert edit_similarity("a\nb", "a\n\nb") == pytest.approx(1 - 1 / 4)
    assert edit_similarity("", "") == 1.0
    # Word sequences differ (missed JOIN) but the char streams match: the boundary
    # decomposition still applies — one ''-vs-' ' boundary, max length 8.
    ratio = edit_similarity("que ries", "queries")
    assert ratio == pytest.approx(1 - 1 / 8)
    # Non-canonical whitespace still classifies per gap: "a \n b" ~ "a\nb".
    assert edit_similarity("a \n b", "a\nb") == 1.0


def test_pk_windowdiff_hand_case() -> None:
    # 4 words, true boundary in the middle: segments of 2 and 2, so k = 2.
    true = [SPACE, PARA, SPACE]
    all_space = [SPACE, SPACE, SPACE]
    near_miss = [PARA, SPACE, SPACE]  # boundary one gap early
    # Missing the boundary entirely: every probe pair disagrees.
    assert pk(true, all_space) == 1.0
    assert windowdiff(true, all_space) == 1.0
    # Near miss: probes (0,2) agree on "different segments", (1,3) disagree.
    assert pk(true, near_miss) == 0.5
    assert windowdiff(true, near_miss) == 0.5


def test_pk_windowdiff_near_miss_beats_no_boundary() -> None:
    true = [SPACE] * 5 + [PARA] + [SPACE] * 5
    near = [SPACE] * 4 + [PARA] + [SPACE] * 6
    none = [SPACE] * 11
    assert 0.0 < pk(true, near) < pk(true, none) <= 1.0
    assert 0.0 < windowdiff(true, near) < windowdiff(true, none) <= 1.0


def test_pk_windowdiff_short_document() -> None:
    # 2 words with one gap: N=2 <= k=2, no probe fits -> NaN (excluded from means),
    # never 0.0, which would count a no-evidence doc as a perfect prediction.
    assert math.isnan(pk([SPACE], [PARA]))
    assert math.isnan(windowdiff([SPACE], [PARA]))
    assert math.isnan(pk([], []))
