import random

import pytest

from newlinefix.corruption import CorruptionConfig, make_example, render_corrupted
from newlinefix.gaps import JOIN, GapText, gaps_to_text, normalize, text_to_gaps


def _random_clean_text(rng: random.Random) -> GapText:
    words = [
        "".join(rng.choice("abcdefgh.,•") for _ in range(rng.randint(1, 10)))
        for _ in range(rng.randint(2, 60))
    ]
    gaps = [rng.choice([1, 2, 3]) for _ in range(len(words) - 1)]
    return GapText(words, gaps)


def test_no_split_is_identity():
    rng = random.Random(0)
    clean = _random_clean_text(rng)
    words, labels = make_example(clean, rng, CorruptionConfig(p_word_split=0.0))
    assert words == clean.words
    assert labels == clean.gaps


def test_true_labels_reconstruct_clean_text():
    """Rendering corrupted words with the TRUE labels must recover the clean text."""
    rng = random.Random(1)
    for _ in range(100):
        clean = _random_clean_text(rng)
        words, labels = make_example(clean, rng, CorruptionConfig(p_word_split=0.3))
        assert gaps_to_text(GapText(words, labels)) == gaps_to_text(clean)


def test_forced_split_produces_join_labels():
    rng = random.Random(2)
    clean = GapText(["queries", "keys"], [1])
    words, labels = make_example(clean, rng, CorruptionConfig(p_word_split=1.0, min_split_len=2))
    assert len(words) == 4
    assert labels.count(JOIN) == 2
    assert words[0] + words[1] == "queries"
    assert words[2] + words[3] == "keys"


def test_short_words_never_split():
    rng = random.Random(3)
    clean = GapText(["a", "b", "c"], [1, 1])
    words, labels = make_example(clean, rng, CorruptionConfig(p_word_split=1.0, min_split_len=1))
    assert words == clean.words
    assert JOIN not in labels


@pytest.mark.parametrize(("p_spurious", "p_keep"), [(0.0, 0.0), (1.0, 1.0), (0.3, 0.3)])
def test_rendered_input_preserves_word_sequence(p_spurious: float, p_keep: float):
    rng = random.Random(4)
    cfg = CorruptionConfig(
        p_word_split=0.2, p_spurious_newline=p_spurious, p_keep_true_newline=p_keep
    )
    for _ in range(30):
        clean = _random_clean_text(rng)
        words, labels = make_example(clean, rng, cfg)
        rendered = render_corrupted(words, labels, rng, cfg)
        assert text_to_gaps(rendered).words == words


def test_end_to_end_shape_matches_readme_example():
    """A corrupted rendering is still normalizable and word-preserving."""
    clean = normalize(
        "3.2.3 Applications of Attention in our Model\n\nThe Transformer uses multi-head "
        'attention in three different ways:\n• In "encoder-decoder attention" layers, '
        "the queries come from the previous decoder layer."
    )
    rng = random.Random(5)
    cfg = CorruptionConfig()
    words, labels = make_example(text_to_gaps(clean), rng, cfg)
    rendered = render_corrupted(words, labels, rng, cfg)
    assert gaps_to_text(GapText(words, labels)) == clean
    assert "".join(text_to_gaps(rendered).words) == "".join(text_to_gaps(clean).words)
