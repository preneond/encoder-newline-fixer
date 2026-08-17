"""Serve the newline-fixer HTTP API with an explicit model choice.

The model source is a CLI argument here (discoverable, listable) and is handed to
the app through the NEWLINEFIX_MODEL_DIR environment variable — the same contract
the Docker image uses, where env is the natural configuration channel.

Usage:
    uv run poe serve                                     # artifacts/encoder
    uv run poe serve --model artifacts/encoder-electra-small
    uv run poe serve --model <user>/newlinefix-encoder   # straight from the HF Hub
    uv run poe serve --list-models
"""

import os
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console

from newlinefix.api import DEFAULT_HUB_MODEL, DEFAULT_LOCAL_MODEL, default_model_source

console = Console()

ARTIFACTS = Path("artifacts")


def local_models() -> list[str]:
    """Servable checkpoints under artifacts/ (dirs written by train_encoder.py)."""
    if not ARTIFACTS.is_dir():
        return []
    return sorted(str(d) for d in ARTIFACTS.iterdir() if (d / "predictor_config.json").is_file())


def main(
    model: Annotated[
        str | None,
        typer.Option(
            help="model to serve: a local artifact dir (see --list-models) or a "
            "Hugging Face Hub repo id; defaults to $NEWLINEFIX_MODEL_DIR, then "
            f"{DEFAULT_LOCAL_MODEL} if present, then {DEFAULT_HUB_MODEL} from the Hub"
        ),
    ] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    list_models: Annotated[
        bool, typer.Option("--list-models", help="list servable local artifacts and exit")
    ] = False,
) -> None:
    """Run the newline-fixer API (uvicorn) with the chosen model."""
    available = local_models()
    if list_models:
        if available:
            console.print("servable local artifacts:")
            for name in available:
                console.print(f"  {name}")
        else:
            console.print("no local artifacts found — train one or use a HF Hub repo id")
        raise typer.Exit()

    chosen = model or os.environ.get("NEWLINEFIX_MODEL_DIR") or default_model_source()
    if not Path(chosen).exists():
        if "/" not in chosen:
            raise SystemExit(
                f"model {chosen!r} is neither a local directory nor a Hub repo id; "
                f"available local artifacts: {available or 'none'}"
            )
        # Could be a Hub repo id — or a typo'd local path. Say so; a bad id fails
        # loudly at startup (the API loads the model in its lifespan hook).
        console.print(
            f"[yellow]{chosen!r} is not a local directory — treating it as a "
            "Hugging Face Hub repo id[/yellow]"
        )
    console.print(f"serving model [bold]{chosen}[/] on http://{host}:{port}")
    if available:
        console.print(f"other local artifacts: {[a for a in available if a != chosen]}")

    os.environ["NEWLINEFIX_MODEL_DIR"] = chosen
    uvicorn.run("newlinefix.api:app", host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
