"""Build canonical train/val/test document JSONL files from streamed corpora.

Streams two sources (encyclopedic wikipedia + markdown), splits documents
deterministically by content hash, and writes {out}/train.jsonl, val.jsonl,
test.jsonl via newlinefix.data.write_documents.

Usage:
    uv run python scripts/prepare_data.py --out data/docs
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import chain
from pathlib import Path

from tqdm import tqdm

from newlinefix.corpora import iter_markdown_docs, iter_wikitext_docs, split_for_text
from newlinefix.data import write_documents
from newlinefix.gaps import NEWLINE, PARA, SPACE, text_to_gaps

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/docs"))
    parser.add_argument("--wikitext-docs", type=int, default=12000)
    parser.add_argument("--markdown-docs", type=int, default=12000)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--test-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_stats(splits: dict[str, list[dict]]) -> None:
    """Per split and source: doc/word counts and gap-class distribution."""
    header = (
        f"{'split':<7}{'source':<16}{'docs':>7}{'words':>10}{'SPACE':>10}{'NEWLINE':>9}{'PARA':>8}"
    )
    print(header)
    print("-" * len(header))
    for split_name in SPLITS:
        by_source: dict[str, list[str]] = {}
        for doc in splits[split_name]:
            by_source.setdefault(doc["source"], []).append(doc["text"])
        for source in sorted(by_source):
            words = 0
            gap_counts: Counter[int] = Counter()
            for text in by_source[source]:
                gap_text = text_to_gaps(text)
                words += len(gap_text.words)
                gap_counts.update(gap_text.gaps)
            print(
                f"{split_name:<7}{source:<16}{len(by_source[source]):>7}{words:>10}"
                f"{gap_counts[SPACE]:>10}{gap_counts[NEWLINE]:>9}{gap_counts[PARA]:>8}"
            )


def main() -> None:
    args = parse_args()
    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    docs = chain(
        tqdm(
            iter_wikitext_docs(args.wikitext_docs, args.seed),
            total=args.wikitext_docs,
            desc="wikipedia",
        ),
        tqdm(
            iter_markdown_docs(args.markdown_docs, args.seed),
            total=args.markdown_docs,
            desc="markdown",
        ),
    )
    for doc in docs:
        splits[split_for_text(doc["text"], args.val_frac, args.test_frac)].append(doc)

    for split_name in SPLITS:
        path = args.out / f"{split_name}.jsonl"
        count = write_documents(path, splits[split_name])
        print(f"wrote {path}: {count} docs")
    print_stats(splits)


if __name__ == "__main__":
    main()
