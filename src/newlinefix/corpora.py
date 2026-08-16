"""Training-corpus acquisition and cleaning into canonical documents.

Two streamed sources (never downloaded in full):
  * ``iter_wikitext_docs`` — encyclopedic prose with headings (see its docstring for
    why it streams wikimedia/wikipedia rather than Salesforce/wikitext).
  * ``iter_markdown_docs`` — markdown-flavored text (headings, bullets) so models see
    the structures in the challenge's arXiv example.

Documents are canonical (``normalize``d): semantic newlines only — "\\n\\n" between
paragraphs and around headings, "\\n" for line-level breaks such as bullet items,
no hard wrapping. Acquisition (network) is separated from pure cleaning helpers so
cleaning is unit-testable offline.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from newlinefix.gaps import SPACE, GapText, gaps_to_text, normalize, text_to_gaps

if TYPE_CHECKING:
    from datasets import IterableDataset

MIN_DOC_WORDS = 80
MIN_STRUCTURAL_GAPS = 2  # gaps that are NEWLINE/PARA: a doc must have some structure
MAX_DOC_WORDS = 3000
MIN_ACCEPTABLE_CHAR_FRACTION = 0.6

_ACCEPTABLE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n.,;:!?'\"()-"
)

_DOLLAR_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_BRACKET_MATH_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)

# ---------------------------------------------------------------------------
# Pure cleaning helpers (offline, unit-testable)
# ---------------------------------------------------------------------------


def remove_code_fences(text: str) -> str:
    """Drop ```-fenced blocks including the fence lines themselves.

    Line-based so info strings (```python) are handled; an unclosed fence drops the
    remainder of the document rather than leaking raw code into the corpus.
    """
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def remove_display_math(text: str) -> str:
    """Drop display-math blocks: ``$$...$$`` and ``\\[...\\]`` (arxiver uses the latter)."""
    return _BRACKET_MATH_RE.sub("", _DOLLAR_MATH_RE.sub("", text))


def clean_markdown(text: str) -> str:
    """Strip non-prose structures from markdown; result still needs ``normalize``."""
    return remove_display_math(remove_code_fences(text))


def acceptable_char_fraction(text: str) -> float:
    """Fraction of characters that are ASCII letters/digits, whitespace, common punct."""
    if not text:
        return 0.0
    return sum(ch in _ACCEPTABLE_CHARS for ch in text) / len(text)


def is_acceptable(text: str) -> bool:
    """Keep canonical docs that are mostly English prose with some line structure."""
    if acceptable_char_fraction(text) < MIN_ACCEPTABLE_CHAR_FRACTION:
        return False
    gap_text = text_to_gaps(text)
    if len(gap_text.words) < MIN_DOC_WORDS:
        return False
    structural = sum(1 for gap in gap_text.gaps if gap != SPACE)
    return structural >= MIN_STRUCTURAL_GAPS


def truncate_words(text: str, max_words: int) -> str:
    """Truncate canonical text to at most ``max_words`` words at a word boundary."""
    gap_text = text_to_gaps(text)
    if len(gap_text.words) <= max_words:
        return text
    return gaps_to_text(GapText(gap_text.words[:max_words], gap_text.gaps[: max_words - 1]))


def build_wikipedia_doc(title: str, body: str) -> str:
    """Assemble one canonical article: title becomes a leading heading paragraph.

    Section headings already sit on their own lines inside the wikipedia ``text``
    field, so ``normalize`` gives them PARA/NEWLINE gaps without extra handling.
    """
    title = title.strip()
    return normalize(f"{title}\n\n{body}" if title else body)


def split_for_text(text: str, val_frac: float, test_frac: float) -> str:
    """Deterministic content-keyed split: identical text always lands in one split."""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2.0**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def _emit_canonical(raw_docs: Iterable[str], source: str, max_docs: int) -> Iterator[dict]:
    """Shared tail of both pipelines: normalize, filter, truncate, exact-dedup."""
    seen: set[str] = set()
    emitted = 0
    for raw in raw_docs:
        if emitted >= max_docs:
            return
        text = normalize(raw)
        if not is_acceptable(text):
            continue
        text = truncate_words(text, MAX_DOC_WORDS)
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        emitted += 1
        yield {"text": text, "source": source}


# ---------------------------------------------------------------------------
# Acquisition (network, streaming)
# ---------------------------------------------------------------------------


def _load_streaming(
    path: str, name: str | None = None, data_dir: str | None = None
) -> IterableDataset:
    # Local import: keep pure helpers importable fast and fully offline.
    from datasets import load_dataset

    return load_dataset(path, name=name, data_dir=data_dir, split="train", streaming=True)


def iter_wikitext_docs(max_docs: int, seed: int) -> Iterator[dict]:
    """Stream encyclopedic documents from wikimedia/wikipedia 20231101.en.

    A streaming probe of Salesforce/wikitext wikitext-103-raw-v1 showed the "raw"
    variant is still word-tokenized: " @-@ " / " @.@ " joiners, spaces before every
    punctuation mark, split contractions ("game 's"), and space-padded quotes. That
    damage cannot be reliably undone by a small detokenizer (quote sides and
    contractions are ambiguous), so the wikipedia dump is streamed instead: the same
    encyclopedic register with natural spacing. The article title is emitted as a
    leading heading paragraph; the ``text`` field already separates paragraphs and
    section headings with newlines.
    """
    dataset = _load_streaming("wikimedia/wikipedia", name="20231101.en")
    shuffled = dataset.shuffle(seed=seed, buffer_size=500)
    raw_docs = (build_wikipedia_doc(row["title"], row["text"]) for row in shuffled)
    yield from _emit_canonical(raw_docs, "wikipedia", max_docs)


#: (hub path, data_dir, text field, source tag) tried in order by iter_markdown_docs.
_MARKDOWN_CANDIDATES: tuple[tuple[str, str | None, str, str], ...] = (
    ("neuralwork/arxiver", None, "markdown", "arxiver"),
    ("bigcode/the-stack-smol", "data/markdown", "content", "stack-markdown"),
)


def _open_markdown_stream(seed: int) -> tuple[IterableDataset, str, str]:
    """Return the first markdown corpus that passes a one-row streaming probe."""
    errors: list[str] = []
    for path, data_dir, field, source in _MARKDOWN_CANDIDATES:
        try:
            dataset = _load_streaming(path, data_dir=data_dir)
            first = next(iter(dataset.take(1)))
            _ = first[field]
        except Exception as exc:  # gated repo, missing field, network failure, ...
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        return dataset.shuffle(seed=seed, buffer_size=200), field, source
    raise RuntimeError("no markdown corpus could be loaded:\n" + "\n".join(errors))


def iter_markdown_docs(max_docs: int, seed: int) -> Iterator[dict]:
    """Stream markdown documents (headings, bullets) as canonical text."""
    dataset, field, source = _open_markdown_stream(seed)
    raw_docs = (clean_markdown(str(row[field])) for row in dataset)
    yield from _emit_canonical(raw_docs, source, max_docs)
