import random

import pytest

from newlinefix.gaps import (
    GAP_STRINGS,
    JOIN,
    NEWLINE,
    PARA,
    SPACE,
    GapText,
    gaps_to_text,
    normalize,
    text_to_gaps,
)

README_EXPECTED = (
    "3.2.3 Applications of Attention in our Model\n"
    "\n"
    "The Transformer uses multi-head attention in three different ways:\n"
    '• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.\n'
    "[...]"
)


def test_classify_basic_separators():
    gap_text = text_to_gaps("a b\nc\n\nd")
    assert gap_text.words == ["a", "b", "c", "d"]
    assert gap_text.gaps == [SPACE, NEWLINE, PARA]


@pytest.mark.parametrize(
    ("text", "expected_gap"),
    [
        ("a\r\nb", NEWLINE),
        ("a \n b", NEWLINE),
        ("a\tb", SPACE),
        ("a  b", SPACE),
        ("a\n\n\n\nb", PARA),
        ("a \n\n b", PARA),
        # Unicode/legacy line boundaries count as line breaks, not spaces.
        ("a\rb", NEWLINE),
        ("a\r\rb", PARA),
        ("a\x85b", NEWLINE),
        ("a b", NEWLINE),  # noqa: RUF001 (U+2028 LINE SEPARATOR is the point)
        ("a b", PARA),  # noqa: RUF001 (U+2029 PARAGRAPH SEPARATOR is the point)
        ("a\x0cb", NEWLINE),
        ("a\r\n\r\nb", PARA),
    ],
)
def test_classify_separator_variants(text: str, expected_gap: int):
    assert text_to_gaps(text).gaps == [expected_gap]


def test_empty_and_single_word():
    assert text_to_gaps("") == GapText([], [])
    assert text_to_gaps("   \n ") == GapText([], [])
    assert text_to_gaps("  hi  ") == GapText(["hi"], [])
    assert gaps_to_text(GapText([], [])) == ""
    assert gaps_to_text(GapText(["hi"], [])) == "hi"


def test_gap_count_invariant_enforced():
    with pytest.raises(ValueError):
        GapText(["a", "b"], [])
    with pytest.raises(ValueError):
        GapText(["a"], [SPACE])


def test_round_trip_on_canonical_text():
    assert normalize(README_EXPECTED) == README_EXPECTED


def test_normalize_is_idempotent():
    messy = "  3.2.3 Applications\n of Attention\n in our Model The\r\nTransformer  uses\n\n\nx "
    once = normalize(messy)
    assert normalize(once) == once


def test_round_trip_random_gap_texts():
    rng = random.Random(0)
    alphabet = "abcXYZ0123.,•\"'-()"
    for _ in range(200):
        n_words = rng.randint(1, 40)
        words = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12))) for _ in range(n_words)
        ]
        gaps = [rng.choice([SPACE, NEWLINE, PARA]) for _ in range(n_words - 1)]
        rendered = gaps_to_text(GapText(words, gaps))
        assert text_to_gaps(rendered) == GapText(words, gaps)


def test_join_renders_as_no_separator():
    assert gaps_to_text(GapText(["que", "ries"], [JOIN])) == "queries"
    assert GAP_STRINGS[JOIN] == ""
