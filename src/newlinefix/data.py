"""Dataset assembly: canonical documents (JSONL) -> corrupted training windows.

Documents are stored one JSON object per line: {"text": <canonical text>, "source": <str>}.
Corpus acquisition lives in ``newlinefix.corpora``; this module is corpus-agnostic.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from newlinefix.corruption import CorruptionConfig, make_example
from newlinefix.gaps import text_to_gaps

Window = tuple[list[str], list[int]]


def read_documents(path: Path | str) -> Iterator[str]:
    """Yield canonical document texts from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]


def write_documents(path: Path | str, docs: Iterable[dict]) -> int:
    """Write {"text", "source"} records as JSONL; returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
    return count


def doc_to_windows(
    text: str, max_words: int, rng: random.Random, cfg: CorruptionConfig
) -> Iterator[Window]:
    """Corrupt one document and chop it into contiguous (words, labels) windows.

    labels[i] is the true gap between words[i] and words[i+1]; the gap that
    straddles a window boundary is dropped rather than duplicated.
    """
    words, labels = make_example(text_to_gaps(text), rng, cfg)
    for start in range(0, len(words), max_words):
        window_words = words[start : start + max_words]
        if len(window_words) < 2:
            continue
        yield window_words, labels[start : start + len(window_words) - 1]


def load_training_windows(
    path: Path | str,
    max_words: int,
    seed: int,
    cfg: CorruptionConfig | None = None,
    limit: int | None = None,
) -> list[Window]:
    """Materialize shuffled training windows from a documents JSONL file.

    Deterministic given (path contents, max_words, seed, cfg). ``limit`` takes a
    random subsample (applied after shuffling), not a prefix of the corpus.
    """
    cfg = cfg or CorruptionConfig()
    rng = random.Random(seed)
    windows: list[Window] = []
    for text in read_documents(path):
        windows.extend(doc_to_windows(text, max_words, rng, cfg))
    rng.shuffle(windows)
    return windows[:limit] if limit is not None else windows
