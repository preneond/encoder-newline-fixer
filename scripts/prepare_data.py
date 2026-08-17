"""Build canonical train/val/test document JSONL files from streamed corpora.

Streams two sources (encyclopedic wikipedia + markdown), splits documents
deterministically by content hash, and writes {out}/train.jsonl, val.jsonl,
test.jsonl via newlinefix.data.write_documents.

Usage:
    uv run python scripts/prepare_data.py --out data/docs
"""

from collections import Counter
from itertools import chain
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from newlinefix.corpora import iter_markdown_docs, iter_wikitext_docs, split_for_text
from newlinefix.data import write_documents
from newlinefix.gaps import NEWLINE, PARA, SPACE, text_to_gaps

SPLITS = ("train", "val", "test")

console = Console()


def stats_table(splits: dict[str, list[dict]]) -> Table:
    """Per split and source: doc/word counts and gap-class distribution."""
    table = Table(title="Corpus statistics")
    table.add_column("split")
    table.add_column("source")
    for column in ("docs", "words", "SPACE", "NEWLINE", "PARA"):
        table.add_column(column, justify="right")
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
            table.add_row(
                split_name,
                source,
                f"{len(by_source[source]):,}",
                f"{words:,}",
                f"{gap_counts[SPACE]:,}",
                f"{gap_counts[NEWLINE]:,}",
                f"{gap_counts[PARA]:,}",
            )
    return table


def main(
    out: Path = Path("data/docs"),
    wikitext_docs: int = 12000,
    markdown_docs: int = 12000,
    val_frac: float = 0.01,
    test_frac: float = 0.01,
    seed: int = 42,
) -> None:
    """Stream, clean, and split the corpora into canonical train/val/test JSONL."""
    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    docs = chain(
        tqdm(iter_wikitext_docs(wikitext_docs, seed), total=wikitext_docs, desc="wikipedia"),
        tqdm(iter_markdown_docs(markdown_docs, seed), total=markdown_docs, desc="markdown"),
    )
    for doc in docs:
        splits[split_for_text(doc["text"], val_frac, test_frac)].append(doc)

    for split_name in SPLITS:
        path = out / f"{split_name}.jsonl"
        count = write_documents(path, splits[split_name])
        console.print(f"wrote [bold]{path}[/]: {count:,} docs")
    console.print(stats_table(splits))


if __name__ == "__main__":
    typer.run(main)
