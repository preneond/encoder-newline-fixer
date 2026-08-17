"""Deploy the demo UI to a Hugging Face Space (challenge deliverable).

A STATIC Space (free tier): space/index.html runs the quantized ONNX model
entirely in the visitor's browser via transformers.js — no server-side compute.
The ONNX export lives in the model repo (onnx/model_quantized.onnx), verified
gap-for-gap identical to the served torch checkpoint on the README example.

Usage:
    uv run poe publish-space          # -> https://huggingface.co/spaces/preneond/newlinefix
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from newlinefix.api import DEFAULT_HUB_MODEL

console = Console()

DEFAULT_SPACE_REPO = "preneond/newlinefix"

SPACE_CARD = f"""---
title: Newline Fixer
emoji: 📝
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
---

# Newline Fixer

Minimal demo UI for the newline-placement fixer. Paste broken text; the model
re-places paragraph breaks, line breaks before bullets and headings, and repairs
words split mid-line — the output words are guaranteed identical to the input.
Inference runs entirely in the browser (quantized ONNX via transformers.js).

Model: [{DEFAULT_HUB_MODEL}](https://huggingface.co/{DEFAULT_HUB_MODEL})
"""


def main(
    repo_id: Annotated[str, typer.Option(help="target Space repo")] = DEFAULT_SPACE_REPO,
    private: Annotated[bool, typer.Option(help="create the Space as private")] = False,
) -> None:
    """Create/update the demo Space: app, requirements, package sources, card."""
    root = Path(__file__).resolve().parent.parent

    # Local import: everything above stays importable/testable offline.
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(
        repo_id, repo_type="space", space_sdk="static", private=private, exist_ok=True
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj=root / "space" / "index.html",
        path_in_repo="index.html",
        commit_message="Space app (in-browser inference)",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj=SPACE_CARD.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Space card",
    )
    console.print(f"deployed Space: {repo_url}")


if __name__ == "__main__":
    typer.run(main)
