"""HTTP service exposing the newline fixer.

POST /fix accepts raw text and returns the same words with re-placed whitespace
(see ``newlinefix.predict.fix_text`` for the guarantee). The served model is
loaded once, lazily, from NEWLINEFIX_MODEL_DIR; when unset, the local artifact
dir is used if it exists, otherwise the published Hub checkpoint is downloaded —
so a fresh clone serves without any local model files. Tests inject a
lightweight predictor via FastAPI dependency overrides instead.

Run locally:  uv run poe serve  (scripts/serve.py picks the model, then runs uvicorn)
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from newlinefix.predict import GapPredictor, fix_text

DEFAULT_LOCAL_MODEL = "artifacts/encoder"
DEFAULT_HUB_MODEL = "preneond/newlinefix-encoder"


def default_model_source() -> str:
    """The local artifact dir when present, else the published Hub repo id."""
    if Path(DEFAULT_LOCAL_MODEL).exists():
        return DEFAULT_LOCAL_MODEL
    return DEFAULT_HUB_MODEL


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the model at startup: a bad model source fails the boot loudly
    instead of surfacing as HTTP 500s, and concurrent first requests can't
    each trigger their own load. (Tests bypass this via dependency overrides.)
    """
    get_predictor()
    yield


app = FastAPI(
    title="Newline Fixer",
    description="Fixes newline placement in English text; output words are "
    "guaranteed identical to the input — only whitespace changes.",
    lifespan=_lifespan,
)


class FixRequest(BaseModel):
    text: str


class FixResponse(BaseModel):
    text: str


@lru_cache(maxsize=1)
def get_predictor() -> GapPredictor:
    """Load the served model once per process."""
    # Local import: the API module stays importable without torch installed.
    from newlinefix.models.encoder import EncoderGapPredictor

    return EncoderGapPredictor.load(
        os.environ.get("NEWLINEFIX_MODEL_DIR") or default_model_source()
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/fix")
def fix(
    request: FixRequest, predictor: Annotated[GapPredictor, Depends(get_predictor)]
) -> FixResponse:
    return FixResponse(text=fix_text(request.text, predictor))
