"""Evaluation metrics for gap classification and newline placement.

All classification metrics operate on aligned gap-label sequences (see
``newlinefix.gaps``); text-level metrics (``edit_similarity``) compare rendered
outputs. Segmentation metrics (``pk``/``windowdiff``) treat PARA gaps as
paragraph-segment boundaries over the word sequence.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence

import numpy as np

from newlinefix.gaps import (
    GAP_LABELS,
    GAP_STRINGS,
    NEWLINE,
    NUM_GAP_CLASSES,
    PARA,
    gaps_to_text,
    text_to_gaps,
)


def confusion_matrix(
    true: list[int], pred: list[int], n_classes: int = NUM_GAP_CLASSES
) -> np.ndarray:
    """Confusion matrix with rows = true class, columns = predicted class."""
    if len(true) != len(pred):
        raise ValueError(f"length mismatch: {len(true)} true vs {len(pred)} pred labels")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(true, dtype=np.int64), np.asarray(pred, dtype=np.int64)), 1)
    return cm


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def per_class_prf(cm: np.ndarray) -> dict[str, dict[str, float]]:
    """Precision/recall/F1/support per gap class, keyed by GAP_LABELS names."""
    out: dict[str, dict[str, float]] = {}
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        precision, recall, f1 = _prf(tp, fp, fn)
        out[GAP_LABELS[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(cm[c, :].sum()),
        }
    return out


def macro_f1(cm: np.ndarray, classes: Sequence[int]) -> float:
    """Unweighted mean F1 over the given class ids."""
    prf = per_class_prf(cm)
    return float(np.mean([prf[GAP_LABELS[c]]["f1"] for c in classes]))


def accuracy(cm: np.ndarray) -> float:
    """Fraction of gaps classified correctly; 0.0 for an empty matrix."""
    total = float(cm.sum())
    return float(np.trace(cm)) / total if total > 0 else 0.0


def break_prf(true: list[int], pred: list[int]) -> dict[str, float]:
    """P/R/F1 for the binary task "is there a line break here at all".

    Positive class: gap in {NEWLINE, PARA}. Measures break *placement* while
    forgiving NEWLINE/PARA confusions.
    """
    if len(true) != len(pred):
        raise ValueError(f"length mismatch: {len(true)} true vs {len(pred)} pred labels")
    t = np.asarray([g in (NEWLINE, PARA) for g in true], dtype=np.bool_)
    p = np.asarray([g in (NEWLINE, PARA) for g in pred], dtype=np.bool_)
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p))
    fn = float(np.sum(t & ~p))
    precision, recall, f1 = _prf(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def _boundary_cumsum(labels: Sequence[int]) -> np.ndarray:
    """cum[j] = number of PARA boundaries among gaps 0..j-1; len = word count."""
    b = np.asarray([1 if g == PARA else 0 for g in labels], dtype=np.int64)
    return np.concatenate(([0], np.cumsum(b)))


def _auto_k(true_cum: np.ndarray) -> int:
    """k = max(2, round(mean true segment length in words / 2))."""
    n_words = len(true_cum)
    n_segments = int(true_cum[-1]) + 1
    return max(2, round(n_words / n_segments / 2))


def pk(true_labels: list[int], pred_labels: list[int]) -> float:
    """Pk segmentation error (Beeferman et al. 1999) over PARA boundaries.

    Words are the atomic units; a segment boundary sits at every gap labeled
    PARA. A probe of width k slides over the words: for each position pair
    (i, i + k), reference and hypothesis are compared on whether the two words
    fall in the same segment. Pk is the disagreement rate::

        Pk = (1 / (N - k)) * sum_{i=0}^{N-k-1} [same_ref(i, i+k) != same_hyp(i, i+k)]

    where N is the word count and k = max(2, round(mean true segment length / 2)).
    Lower is better; 0.0 means every probe agrees. Returns 0.0 when N <= k
    (document too short to place a probe).
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("pk: label sequences must have equal length")
    cum_t = _boundary_cumsum(true_labels)
    cum_p = _boundary_cumsum(pred_labels)
    k = _auto_k(cum_t)
    if len(cum_t) <= k:
        return 0.0
    # cum doubles as a segment id per word: equal cum values <=> same segment.
    same_t = cum_t[:-k] == cum_t[k:]
    same_p = cum_p[:-k] == cum_p[k:]
    return float(np.mean(same_t != same_p))


def windowdiff(true_labels: list[int], pred_labels: list[int]) -> float:
    """WindowDiff segmentation error (Pevzner & Hearst 2002) over PARA boundaries.

    Like ``pk`` but compares boundary *counts* inside each probe window, which
    penalizes false positives and near misses more evenly::

        WD = (1 / (N - k)) * sum_{i=0}^{N-k-1} [b_ref(i, i+k) != b_hyp(i, i+k)]

    where b(i, i+k) counts boundaries strictly between words i and i+k, N is
    the word count, and k = max(2, round(mean true segment length / 2)).
    Lower is better. Returns 0.0 when N <= k.
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("windowdiff: label sequences must have equal length")
    cum_t = _boundary_cumsum(true_labels)
    cum_p = _boundary_cumsum(pred_labels)
    k = _auto_k(cum_t)
    if len(cum_t) <= k:
        return 0.0
    counts_t = cum_t[k:] - cum_t[:-k]
    counts_p = cum_p[k:] - cum_p[:-k]
    return float(np.mean(counts_t != counts_p))


def _levenshtein(a: str, b: str) -> int:
    """Unit-cost edit distance (insert/delete/substitute) via the classic DP."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[len(b)]


# 4x4 edit distances between the gap separator strings '', ' ', '\n', '\n\n'.
_SEP_DIST = [[_levenshtein(a, b) for b in GAP_STRINGS] for a in GAP_STRINGS]


def edit_similarity(pred_text: str, true_text: str) -> float:
    """Character-level similarity in [0, 1]: 1 - editdist / max(len).

    Since prediction only rewrites whitespace, the word sequences normally
    match; then the exact edit distance decomposes into a sum over aligned
    gap pairs of separator edit distances (each separator is one of '', ' ',
    '\\n', '\\n\\n'), with texts taken in canonical (normalized) form. If the
    word sequences differ (e.g. a JOIN mistake merged two words), falls back
    to ``difflib.SequenceMatcher.ratio``.
    """
    pred = text_to_gaps(pred_text)
    true = text_to_gaps(true_text)
    if pred.words != true.words:
        return difflib.SequenceMatcher(None, pred_text, true_text).ratio()
    dist = sum(_SEP_DIST[p][t] for p, t in zip(pred.gaps, true.gaps, strict=True))
    denom = max(len(gaps_to_text(pred)), len(gaps_to_text(true)))
    if denom == 0:
        return 1.0
    return 1.0 - dist / denom
