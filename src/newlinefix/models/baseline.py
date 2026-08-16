"""Trivial baselines: what you get without learning anything.

Both baselines predict each gap independently from at most the two adjacent
words, so windowing is unnecessary; ``max_words`` is set high enough that
``predict_gaps`` always takes the single-window path.
"""

from __future__ import annotations

import re

from newlinefix.gaps import NEWLINE, PARA, SPACE
from newlinefix.predict import GapPredictor

# Standalone bullet markers that start a new list line (en dash is intentional).
_BULLET_MARKERS = frozenset({"•", "●", "▪", "‣", "◦", "-", "–", "*"})  # noqa: RUF001

# Numbered section headings like "3.2.3" or "1.4." (at least two numeric parts).
_SECTION_RE = re.compile(r"^\d+(\.\d+)+\.?$")


class AllSpaceBaseline(GapPredictor):
    """Majority-class baseline: every gap is a single space."""

    max_words = 1_000_000_000

    def predict_window(self, words: list[str]) -> list[int]:
        return [SPACE] * (len(words) - 1)


class RuleBaseline(GapPredictor):
    """Hand-written rules on the word right of each gap; the interpretable foil."""

    max_words = 1_000_000_000

    def predict_window(self, words: list[str]) -> list[int]:
        return [self._classify(words[i + 1]) for i in range(len(words) - 1)]

    @staticmethod
    def _classify(right: str) -> int:
        # Rule 1: line break before a standalone bullet marker.
        if right in _BULLET_MARKERS:
            return NEWLINE
        # Rule 2: paragraph break before a numbered-section token (e.g. "3.2.3").
        if _SECTION_RE.match(right):
            return PARA
        # Rule 3: default to a plain space; never JOIN (merging words is riskier).
        return SPACE
