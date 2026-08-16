from newlinefix.gaps import NEWLINE, PARA, SPACE
from newlinefix.models.baseline import AllSpaceBaseline, RuleBaseline

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


def test_rule_baseline_newline_before_bullet():
    labels = RuleBaseline().predict_gaps(README_WORDS)
    bullet_gap = README_WORDS.index("•") - 1
    assert labels[bullet_gap] == NEWLINE


def test_rule_baseline_para_before_numbered_section():
    labels = RuleBaseline().predict_gaps(README_WORDS)
    section_gap = README_WORDS.index("3.2.3") - 1
    assert labels[section_gap] == PARA


def test_rule_baseline_plain_prose_all_space():
    words = ["the", "queries", "come", "from", "the", "previous", "decoder", "layer"]
    assert RuleBaseline().predict_gaps(words) == [SPACE] * (len(words) - 1)


def test_all_space_baseline_all_space():
    assert AllSpaceBaseline().predict_gaps(README_WORDS) == [SPACE] * (len(README_WORDS) - 1)


def test_long_input_label_count_invariant():
    # >2000 words: the huge max_words must keep this on the single-window path.
    words = ["word"] * 2500
    for predictor in (AllSpaceBaseline(), RuleBaseline()):
        labels = predictor.predict_gaps(words)
        assert len(labels) == len(words) - 1
        assert labels == [SPACE] * (len(words) - 1)
