"""HTTP service exposing the newline fixer.

POST /fix accepts raw text and returns the same words with re-placed whitespace
(see ``newlinefix.predict.fix_text`` for the guarantee). The served model is
loaded once, lazily, from NEWLINEFIX_MODEL_DIR (default: artifacts/encoder);
tests inject a lightweight predictor via FastAPI dependency overrides instead.

Run locally:  uv run poe serve  (scripts/serve.py picks the model, then runs uvicorn)
"""

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from newlinefix.predict import GapPredictor, fix_text

app = FastAPI(
    title="Newline Fixer",
    description="Fixes newline placement in English text; output words are "
    "guaranteed identical to the input — only whitespace changes.",
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

    return EncoderGapPredictor.load(os.environ.get("NEWLINEFIX_MODEL_DIR", "artifacts/encoder"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/fix")
def fix(
    request: FixRequest, predictor: Annotated[GapPredictor, Depends(get_predictor)]
) -> FixResponse:
    return FixResponse(text=fix_text(request.text, predictor))
