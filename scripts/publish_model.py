"""Publish a trained encoder artifact to the Hugging Face Hub.

Uploads the artifact directory (HF-format checkpoint + tokenizer +
predictor_config.json) and a generated model card. Once published, the repo id
works everywhere a local artifact dir does — EncoderGapPredictor.load,
--encoder-dir / --extra in evaluate.py, and NEWLINEFIX_MODEL_DIR for the API —
so the service can be deployed without shipping model files.

Requires a Hub token (huggingface-cli login, or HF_TOKEN in the environment).

Usage:
    uv run poe publish --repo-id <user>/newlinefix-encoder
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

console = Console()


def build_model_card(repo_id: str, config: dict[str, Any]) -> str:
    """Model card for the Hub repo, with validation metrics from the artifact."""
    base_model = config.get("model_name", "distilroberta-base")
    val_metrics = config.get("val_metrics", {})
    metric_rows = "\n".join(
        f"| {name} | {value:.4f} |"
        for name, value in val_metrics.items()
        if isinstance(value, float)
    )
    return f"""---
library_name: transformers
pipeline_tag: token-classification
base_model: {base_model}
tags:
- newline-restoration
- text-segmentation
---

# Newline Fixer — gap-classification encoder

Fine-tuned `{base_model}` that classifies the separator ("gap") between each pair of
consecutive words into JOIN / SPACE / NEWLINE / PARA. Reconstructing text from words
plus predicted gaps restores newline placement with a hard guarantee: the output
words are identical to the input — only whitespace changes.

Trained self-supervised on Wikipedia + arXiv-markdown text whose newline structure
was programmatically destroyed. Full methodology, evaluation protocol, and results:
the project's `report.md`.

## Serving

The raw checkpoint is a standard token-classification model, but gap decoding
(last-subtoken alignment, sliding-window stitching, reconstruction) lives in the
`newlinefix` package:

```python
from newlinefix.models.encoder import EncoderGapPredictor
from newlinefix.predict import fix_text

predictor = EncoderGapPredictor.load("{repo_id}")
print(fix_text("the que\\nries come from here", predictor))
```

The bundled HTTP service serves this repo directly: `NEWLINEFIX_MODEL_DIR={repo_id}`.

## Validation metrics

Macro-F1 is over the structural classes {{JOIN, NEWLINE, PARA}}.

| metric | value |
|---|---|
{metric_rows}
"""


def main(
    repo_id: Annotated[str, typer.Option(help="target Hub repo, e.g. <user>/newlinefix-encoder")],
    artifact_dir: Annotated[Path, typer.Option(help="trained encoder artifact to upload")] = Path(
        "artifacts/encoder"
    ),
    private: Annotated[bool, typer.Option(help="create the repo as private")] = True,
    commit_message: str = "Upload newline-fixer encoder",
) -> None:
    """Upload a trained encoder artifact and its model card to the Hugging Face Hub."""
    config_path = artifact_dir / "predictor_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"{config_path} not found — point --artifact-dir at a directory written by "
            "scripts/train_encoder.py"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    card = build_model_card(repo_id, config)

    # Local import: everything above stays importable/testable offline.
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=artifact_dir, commit_message=commit_message)
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Model card",
    )
    console.print(f"published [bold]{artifact_dir}[/] to {repo_url}")
    console.print(f"serve it directly: NEWLINEFIX_MODEL_DIR={repo_id} uv run poe serve")


if __name__ == "__main__":
    typer.run(main)
