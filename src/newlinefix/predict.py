"""Predictor interface and sliding-window inference over long texts.

Models implement ``predict_window`` for a bounded number of words; ``fix_text``
handles normalization, windowing with overlap, and reconstruction. Overlapping
windows are stitched by trusting each window only on its central region, where
the model has bidirectional context.
"""

from abc import ABC, abstractmethod

from newlinefix.gaps import GapText, gaps_to_text, text_to_gaps


class GapPredictor(ABC):
    """A model that classifies the gap after each word in a bounded window."""

    #: Maximum window size in words accepted by ``predict_window``.
    max_words: int = 200
    #: Words shared between consecutive windows (must be < max_words, even).
    overlap: int = 64

    @abstractmethod
    def predict_window(self, words: list[str]) -> list[int]:
        """Return one gap class per consecutive word pair: len(words) - 1 labels."""

    def predict_gaps(self, words: list[str]) -> list[int]:
        """Predict all len(words) - 1 gap labels, windowing if needed."""
        n = len(words)
        if n < 2:
            return []
        if n <= self.max_words:
            return self._checked_window(words)

        step = self.max_words - self.overlap
        margin = self.overlap // 2
        if step < 1 or margin < 1:
            raise ValueError(
                f"invalid windowing: max_words={self.max_words}, overlap={self.overlap}"
            )
        labels: list[int] = []
        start = 0
        while True:
            end = min(start + self.max_words, n)
            window_labels = self._checked_window(words[start:end])
            # Gap start+i sits between words start+i and start+i+1. Take labels up
            # to `margin` gaps before an interior window's right edge (where the
            # model lacks right context); the next window re-predicts them with
            # bidirectional context. len(labels) is the next global gap to fill.
            last_taken = n - 2 if end == n else end - margin - 2
            labels.extend(window_labels[len(labels) - start : last_taken - start + 1])
            if end == n:
                break
            start += step
        assert len(labels) == n - 1, "windowing left uncovered gaps"
        return labels

    def _checked_window(self, words: list[str]) -> list[int]:
        labels = self.predict_window(words)
        if len(labels) != len(words) - 1:
            raise ValueError(
                f"{type(self).__name__}.predict_window returned {len(labels)} labels "
                f"for {len(words)} words (expected {len(words) - 1})"
            )
        return labels


def fix_text(text: str, predictor: GapPredictor) -> str:
    """Re-place newlines in ``text``: existing (unreliable) whitespace is discarded
    and every gap is re-predicted from the word sequence alone."""
    words = text_to_gaps(text).words
    return gaps_to_text(GapText(words, predictor.predict_gaps(words)))
