"""Offline unit tests for corpus cleaning helpers — no network, no HF hub."""

from newlinefix.corpora import (
    MAX_DOC_WORDS,
    MIN_DOC_WORDS,
    _emit_canonical,
    acceptable_char_fraction,
    build_wikipedia_doc,
    clean_markdown,
    is_acceptable,
    remove_code_fences,
    remove_display_math,
    split_for_text,
    truncate_words,
)
from newlinefix.gaps import NEWLINE, PARA, SPACE, normalize, text_to_gaps


def make_prose(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


def acceptable_doc(n_words: int = MIN_DOC_WORDS) -> str:
    """Canonical doc with enough words and two PARA gaps."""
    third = n_words // 3
    return "\n\n".join(make_prose(k) for k in (third, third, n_words - 2 * third))


class TestRemoveCodeFences:
    def test_removes_fenced_block_and_fence_lines(self) -> None:
        text = "intro text\n```python\nx = 1\nprint(x)\n```\noutro text"
        assert remove_code_fences(text) == "intro text\noutro text"

    def test_unclosed_fence_drops_remainder(self) -> None:
        text = "keep this\n```\nsecret code\nmore code"
        assert remove_code_fences(text) == "keep this"

    def test_multiple_blocks(self) -> None:
        text = "a\n```\ncode1\n```\nb\n```js\ncode2\n```\nc"
        assert remove_code_fences(text) == "a\nb\nc"

    def test_no_fences_is_identity(self) -> None:
        text = "plain\ntext\n\nwith paragraphs"
        assert remove_code_fences(text) == text


class TestRemoveDisplayMath:
    def test_removes_dollar_blocks(self) -> None:
        # Blocks are replaced by a space (not ""), so flanking words never splice.
        text = "before $$x^2 +\ny^2$$ after"
        assert remove_display_math(text) == "before   after"

    def test_removes_bracket_blocks(self) -> None:
        text = "loss is\n\n\\[\\mathcal{L} = a + b \\tag{3}\\]\n\nwhere a is"
        assert remove_display_math(text) == "loss is\n\n \n\nwhere a is"

    def test_unpaired_opener_does_not_eat_prose(self) -> None:
        # An unpaired $$ must not swallow paragraphs up to an unrelated closer.
        text = "broken $$ formula\n\nplain prose here\n\nmore $$x$$ math"
        cleaned = remove_display_math(text)
        assert "plain prose here" in cleaned
        assert "x" not in cleaned.split("more")[1]

    def test_words_not_spliced_across_removed_math(self) -> None:
        assert "wordafter" not in remove_display_math("word$$x$$after")

    def test_keeps_inline_math(self) -> None:
        text = "the rate \\(\\alpha\\) is small"
        assert remove_display_math(text) == text


class TestCleanMarkdown:
    def test_headings_and_bullets_survive_cleaning(self) -> None:
        md = (
            "# Attention\n\n"
            "The Transformer uses attention in three ways:\n"
            "* encoder-decoder attention\n"
            "* self-attention in the encoder\n\n"
            "```python\nignore_me()\n```\n\n"
            "$$a = b$$\n\n"
            "Closing paragraph."
        )
        canonical = normalize(clean_markdown(md))
        assert "ignore_me" not in canonical
        assert "a = b" not in canonical
        gaps = text_to_gaps(canonical).gaps
        assert NEWLINE in gaps  # bullet items keep line-level breaks
        assert PARA in gaps  # heading/paragraph structure kept
        assert "# Attention\n\nThe Transformer" in canonical


class TestAcceptability:
    def test_english_doc_accepted(self) -> None:
        assert is_acceptable(acceptable_doc())

    def test_char_fraction_bounds(self) -> None:
        assert acceptable_char_fraction("") == 0.0
        assert acceptable_char_fraction("plain english text.") == 1.0
        assert acceptable_char_fraction("αβγδ") == 0.0

    def test_rejects_non_ascii_heavy_doc(self) -> None:
        words = " ".join("αβγδεζ" for _ in range(MIN_DOC_WORDS))
        assert not is_acceptable(f"{words}\n\nend\n\nhere")

    def test_rejects_short_doc(self) -> None:
        assert not is_acceptable("short\n\ndoc\n\nhere")

    def test_rejects_doc_without_line_structure(self) -> None:
        assert not is_acceptable(make_prose(2 * MIN_DOC_WORDS))  # only SPACE gaps


class TestTruncateWords:
    def test_truncates_at_word_boundary(self) -> None:
        text = acceptable_doc(120)
        truncated = truncate_words(text, 10)
        words = text_to_gaps(truncated).words
        assert words == text_to_gaps(text).words[:10]
        assert normalize(truncated) == truncated  # still canonical
        assert not truncated[-1].isspace()

    def test_short_text_unchanged(self) -> None:
        text = acceptable_doc(90)
        assert truncate_words(text, MAX_DOC_WORDS) == text

    def test_preserves_gap_classes_in_prefix(self) -> None:
        text = "a b\nc\n\nd e f"
        truncated = truncate_words(text, 4)
        assert truncated == "a b\nc\n\nd"


class TestBuildWikipediaDoc:
    def test_title_becomes_leading_heading_paragraph(self) -> None:
        doc = build_wikipedia_doc("Albedo", "Albedo is a fraction.\n\nSurface albedo is defined.")
        assert doc == "Albedo\n\nAlbedo is a fraction.\n\nSurface albedo is defined."

    def test_section_heading_lines_kept_as_paragraphs(self) -> None:
        # wikimedia/wikipedia leaves section headings on their own lines,
        # often with a trailing space before the paragraph break.
        body = "Intro paragraph.\n\nHistory \n\nThe earliest ancestor."
        doc = build_wikipedia_doc("A", body)
        assert doc == "A\n\nIntro paragraph.\n\nHistory\n\nThe earliest ancestor."
        gaps = text_to_gaps(doc).gaps
        assert gaps.count(PARA) == 3  # title|intro, intro|heading, heading|body

    def test_empty_title_omitted(self) -> None:
        assert build_wikipedia_doc("  ", "Body text here.") == "Body text here."


class TestEmitCanonical:
    def test_filters_dedups_truncates_and_caps(self) -> None:
        good = acceptable_doc()
        long_doc = acceptable_doc(MAX_DOC_WORDS + 50)
        raw = [good, "too short", good, long_doc, acceptable_doc(100), acceptable_doc(110)]
        docs = list(_emit_canonical(raw, "src", max_docs=4))
        texts = [doc["text"] for doc in docs]
        assert len(docs) == 4  # dup and junk dropped, cap respected
        assert len(set(texts)) == 4
        assert all(d["source"] == "src" for d in docs)
        assert max(len(text_to_gaps(t).words) for t in texts) <= MAX_DOC_WORDS
        assert all(normalize(t) == t for t in texts)


class TestSplitForText:
    def test_deterministic_and_content_stable(self) -> None:
        for i in range(50):
            text = f"document number {i} with some words"
            assert split_for_text(text, 0.1, 0.1) == split_for_text(text, 0.1, 0.1)

    def test_zero_fractions_send_everything_to_train(self) -> None:
        assert all(split_for_text(f"doc {i}", 0.0, 0.0) == "train" for i in range(200))

    def test_fractions_roughly_respected(self) -> None:
        counts = {"train": 0, "val": 0, "test": 0}
        for i in range(2000):
            counts[split_for_text(f"synthetic doc {i}", 0.1, 0.1)] += 1
        assert 120 <= counts["val"] <= 280
        assert 120 <= counts["test"] <= 280
        assert counts["train"] == 2000 - counts["val"] - counts["test"]

    def test_growing_test_frac_keeps_membership(self) -> None:
        # Content-keyed: a doc in test at a small fraction stays in test at a
        # larger one, so split membership is stable as the corpus grows.
        for i in range(300):
            text = f"stability doc {i}"
            if split_for_text(text, 0.0, 0.05) == "test":
                assert split_for_text(text, 0.0, 0.2) == "test"


class TestSpaceGapDominates:
    def test_acceptable_doc_gap_profile(self) -> None:
        gaps = text_to_gaps(acceptable_doc()).gaps
        assert gaps.count(SPACE) > gaps.count(PARA) > 0
