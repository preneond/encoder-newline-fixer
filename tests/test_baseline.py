import pytest

from newlinefix.gaps import NEWLINE, PARA, SPACE
from newlinefix.models.baseline import AllSpaceBaseline, RuleBaseline
from newlinefix.predict import GapPredictor

# Word sequence from the README example, with a numbered section token mid-sequence.
README_WORDS = [
    "attention",
    "in",
    "three",
    "different",
    "ways:",
    "•",
    "In",
    "encoder-decoder",
    "attention",
    "layers.",
    "3.2.3",
    "Applications",
    "of",
    "Attention",
    "in",
    "our",
    "Model",
]


@pytest.fixture(scope="module")
def rule_labels() -> list[int]:
    return RuleBaseline().predict_gaps(README_WORDS)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        pytest.param("•", NEWLINE, id="newline-before-bullet"),
        pytest.param("3.2.3", PARA, id="para-before-numbered-section"),
    ],
)
def test_rule_baseline_break_markers(rule_labels: list[int], marker: str, expected: int) -> None:
    assert rule_labels[README_WORDS.index(marker) - 1] == expected


def test_rule_baseline_plain_prose_all_space() -> None:
    words = ["the", "queries", "come", "from", "the", "previous", "decoder", "layer"]
    assert RuleBaseline().predict_gaps(words) == [SPACE] * (len(words) - 1)


def test_all_space_baseline_all_space() -> None:
    assert AllSpaceBaseline().predict_gaps(README_WORDS) == [SPACE] * (len(README_WORDS) - 1)


@pytest.mark.parametrize(
    "predictor", [AllSpaceBaseline(), RuleBaseline()], ids=lambda p: type(p).__name__
)
def test_long_input_label_count_invariant(predictor: GapPredictor) -> None:
    # >2000 words: the huge max_words must keep this on the single-window path.
    words = ["word"] * 2500
    labels = predictor.predict_gaps(words)
    assert len(labels) == len(words) - 1
    assert labels == [SPACE] * (len(words) - 1)
