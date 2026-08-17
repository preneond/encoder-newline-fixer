"""Publish the generated corpus (train/val/test JSONL) to the Hub as a dataset.

Private by default, deliberately: roughly half the corpus is full-text arXiv
papers (via neuralwork/arxiver) whose per-paper licenses often do not permit
redistribution — review that before ever flipping the repo public. The
Wikipedia half is CC BY-SA 4.0. The generated dataset card records both.

Usage:
    uv run poe publish-data                     # -> preneond/newlinefix-corpus (private)
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

DEFAULT_DATASET_REPO = "preneond/newlinefix-corpus"
SPLITS = ("train", "val", "test")


def split_stats(data_dir: Path) -> dict[str, dict[str, int]]:
    """Per split: document count and per-source counts (one JSON object per line)."""
    stats: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts: dict[str, int] = {}
        with open(data_dir / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    source = json.loads(line)["source"]
                    counts[source] = counts.get(source, 0) + 1
        stats[split] = counts
    return stats


def build_dataset_card(repo_id: str, stats: dict[str, dict[str, int]]) -> str:
    rows = "\n".join(
        f"| {split} | {sum(counts.values()):,} | "
        + ", ".join(f"{source}: {n:,}" for source, n in sorted(counts.items()))
        + " |"
        for split, counts in stats.items()
    )
    return f"""---
pretty_name: Newline Fixer Corpus
language:
- en
task_categories:
- token-classification
size_categories:
- 10K<n<100K
---

# Newline Fixer Corpus

Canonical training documents for the newline-placement fixer: English prose with
semantic newlines only — `\\n\\n` between paragraphs and around headings, `\\n` for
line-level breaks such as bullet items, single spaces otherwise. One JSON object
per line: `{{"text": <canonical text>, "source": <corpus tag>}}`.

Documents were streamed from the source corpora, cleaned (code fences, display
math, and HTML removed), normalized, filtered (>=80 words, >=60% English-prose
characters, >=2 structural gaps), exact-deduplicated, and split **by content
hash**, so train/val/test membership is stable. Training pairs are derived
self-supervised: newline structure is destroyed programmatically and the clean
text is the label.

| split | documents | by source |
|---|---|---|
{rows}

## Loading

```python
from datasets import load_dataset

corpus = load_dataset("{repo_id}")
```

## Provenance and licensing

- **wikimedia/wikipedia 20231101.en** — article text is CC BY-SA 4.0.
- **neuralwork/arxiver** — arXiv papers converted to markdown; the underlying
  papers carry per-paper licenses and arxiver's own terms apply.

Because of the arXiv-derived half, this dataset is published **private by
default**; review the source licenses before redistributing or making it public.
"""


def main(
    repo_id: Annotated[str, typer.Option(help="target Hub dataset repo")] = DEFAULT_DATASET_REPO,
    data_dir: Annotated[
        Path, typer.Option(help="directory with train/val/test.jsonl")
    ] = Path("data/docs"),
    private: Annotated[
        bool, typer.Option(help="create the repo as private (see licensing note)")
    ] = True,
    commit_message: str = "Upload newline-fixer corpus",
) -> None:
    """Upload the generated corpus and its dataset card to the Hugging Face Hub."""
    missing = [split for split in SPLITS if not (data_dir / f"{split}.jsonl").exists()]
    if missing:
        raise SystemExit(
            f"missing {missing} in {data_dir} — generate the corpus with "
            "scripts/prepare_data.py first"
        )
    console.print("counting documents per split...")
    stats = split_stats(data_dir)
    card = build_dataset_card(repo_id, stats)

    # Local import: everything above stays importable/testable offline.
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for split in SPLITS:
        console.print(f"uploading {split}.jsonl...")
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=data_dir / f"{split}.jsonl",
            path_in_repo=f"{split}.jsonl",
            commit_message=f"{commit_message} ({split})",
        )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Dataset card",
    )
    console.print(f"published corpus to {repo_url}")


if __name__ == "__main__":
    typer.run(main)
