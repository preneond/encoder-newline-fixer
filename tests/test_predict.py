import pytest

from newlinefix.gaps import NUM_GAP_CLASSES, SPACE
from newlinefix.predict import GapPredictor, fix_text


class IndexOracle(GapPredictor):
    """Labels each global gap i as i % NUM_GAP_CLASSES, decoded from word names.

    Words must be named "w{i}" so the oracle knows each gap's global index; this
    makes stitching errors (wrong window offsets, uncovered gaps) visible.
    """

    max_words = 10
    overlap = 4

    def __init__(self) -> None:
        self.window_sizes: list[int] = []

    def predict_window(self, words: list[str]) -> list[int]:
        self.window_sizes.append(len(words))
        return [int(w[1:]) % NUM_GAP_CLASSES for w in words[:-1]]


@pytest.mark.parametrize("n_words", [2, 9, 10, 11, 14, 15, 16, 17, 20, 37, 100])
def test_windowed_prediction_matches_oracle(n_words: int):
    predictor = IndexOracle()
    words = [f"w{i}" for i in range(n_words)]
    labels = predictor.predict_gaps(words)
    assert labels == [i % NUM_GAP_CLASSES for i in range(n_words - 1)]
    assert all(size <= predictor.max_words for size in predictor.window_sizes)


def test_short_inputs():
    predictor = IndexOracle()
    assert predictor.predict_gaps([]) == []
    assert predictor.predict_gaps(["w0"]) == []


def test_wrong_label_count_raises():
    class Broken(GapPredictor):
        def predict_window(self, words: list[str]) -> list[int]:
            return []

    with pytest.raises(ValueError, match="expected"):
        Broken().predict_gaps(["a", "b"])


def test_invalid_window_config_raises():
    class Misconfigured(IndexOracle):
        max_words = 4
        overlap = 4

    with pytest.raises(ValueError, match="invalid windowing"):
        Misconfigured().predict_gaps([f"w{i}" for i in range(50)])


class AllSpace(GapPredictor):
    def predict_window(self, words: list[str]) -> list[int]:
        return [SPACE] * (len(words) - 1)


def test_fix_text_replaces_whitespace_only():
    assert fix_text("que\nries come\n from here", AllSpace()) == "que ries come from here"
    assert fix_text("", AllSpace()) == ""
    assert fix_text("  word \n ", AllSpace()) == "word"
