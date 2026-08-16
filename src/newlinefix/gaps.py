"""Canonical text representation: a sequence of words separated by classified gaps.

Newline fixing is framed as classifying the separator ("gap") between each pair of
consecutive words. Reconstructing text from words + predicted gaps guarantees the
model can never alter, drop, or hallucinate words — only whitespace changes.

Gap classes:
    JOIN    — no separator; the two tokens are halves of one word ("que" + "ries")
    SPACE   — single space (normal intra-line separator)
    NEWLINE — single "\\n" (line break: bullet items, headings, list entries)
    PARA    — "\\n\\n" (paragraph break)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

JOIN = 0
SPACE = 1
NEWLINE = 2
PARA = 3

GAP_LABELS = ("JOIN", "SPACE", "NEWLINE", "PARA")
GAP_STRINGS = ("", " ", "\n", "\n\n")
NUM_GAP_CLASSES = 4

_WORD_RE = re.compile(r"\S+")


@dataclass
class GapText:
    """Words plus the gap class between each consecutive pair.

    Invariant: len(gaps) == max(len(words) - 1, 0); words contain no whitespace.
    """

    words: list[str]
    gaps: list[int]

    def __post_init__(self) -> None:
        expected = max(len(self.words) - 1, 0)
        if len(self.gaps) != expected:
            raise ValueError(
                f"expected {expected} gaps for {len(self.words)} words, got {len(self.gaps)}"
            )


# Unicode line boundaries normalized to "\n" before counting; U+2029 is a paragraph
# separator, hence "\n\n". "\r\n" is collapsed first so it counts once.
_LINE_BREAKS = str.maketrans(
    {"\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x85": "\n", "\u2028": "\n", "\u2029": "\n\n"}
)


def classify_separator(sep: str) -> int:
    """Map a raw whitespace separator to a gap class by its line-break count."""
    newlines = sep.replace("\r\n", "\n").translate(_LINE_BREAKS).count("\n")
    if newlines == 0:
        return SPACE
    if newlines == 1:
        return NEWLINE
    return PARA


def text_to_gaps(text: str) -> GapText:
    """Split text into words and classify the whitespace between them.

    Leading/trailing whitespace is dropped; JOIN never occurs here because raw
    text cannot contain an empty separator between two words.
    """
    spans = [m.span() for m in _WORD_RE.finditer(text)]
    words = [text[a:b] for a, b in spans]
    gaps = [classify_separator(text[spans[i][1] : spans[i + 1][0]]) for i in range(len(spans) - 1)]
    return GapText(words, gaps)


def gaps_to_text(gap_text: GapText) -> str:
    """Render words joined by their gap separators."""
    if not gap_text.words:
        return ""
    parts = [gap_text.words[0]]
    for gap, word in zip(gap_text.gaps, gap_text.words[1:], strict=True):
        parts.append(GAP_STRINGS[gap])
        parts.append(word)
    return "".join(parts)


def normalize(text: str) -> str:
    """Canonical form: single spaces, single "\\n", or "\\n\\n"; no edge whitespace.

    Idempotent, and ``text_to_gaps``/``gaps_to_text`` round-trip exactly on its output.
    """
    return gaps_to_text(text_to_gaps(text))
