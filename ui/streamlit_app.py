"""Minimal Streamlit UI for the newline-fixing models.

Run locally:  uv run streamlit run ui/streamlit_app.py
Models are detected from their artifact directories; baselines are always available.
"""

import os
import time
from pathlib import Path

import httpx
import streamlit as st

from newlinefix.predict import GapPredictor, fix_text

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Two modes, one app. With NEWLINEFIX_API_URL set (how docker compose runs it) the UI
# is a thin client of the FastAPI service: one model — whatever the API serves — and no
# artifacts or torch needed in the UI image. Unset (local dev), predictors are loaded
# in-process from artifacts/ so the picker can compare encoder, scratch and baselines.
API_URL = os.environ.get("NEWLINEFIX_API_URL")
API_TIMEOUT_S = 60.0

README_EXAMPLE = (
    "3.2.3 Applications of Attention\n"
    " in our Model The Transformer uses multi-head attention in three different ways:"
    ' • In "encoder-decoder attention" layers,\n'
    " the que\n"
    "ries come from the previous decoder layer.[...]"
)

MODEL_HELP = {
    "api": "served model via API (`POST /fix`)",
    "encoder": "fine-tuned distilroberta-base token-gap classifier (the served model)",
    "scratch": "byte-level BiLSTM (~2.4M params) trained from scratch on our data",
    "rules": "hand-written rules: newline before bullets, break before numbered headings",
    "majority": "always predicts a plain space (the trivial baseline)",
}


def _describe(kind: str, directory: Path) -> str:
    """Help text for an auto-discovered artifact dir, from its saved config."""
    import json

    if kind == "encoder":
        cfg = json.loads((directory / "predictor_config.json").read_text(encoding="utf-8"))
        return f"fine-tuned `{cfg.get('model_name', '?')}` token-gap classifier"
    cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if cfg.get("teacher"):
        return f"byte-level BiLSTM distilled from `{cfg['teacher']}`"
    return "byte-level BiLSTM trained from scratch on our data"


def discover_models() -> dict[str, tuple[str, Path | None]]:
    """name -> (kind, artifact dir). Any artifacts/*/ dir with a known config counts."""
    models: dict[str, tuple[str, Path | None]] = {}
    if ARTIFACTS.is_dir():
        for directory in sorted(ARTIFACTS.iterdir()):
            if (directory / "predictor_config.json").exists():
                models[directory.name] = ("encoder", directory)
            elif (directory / "config.json").exists() and (directory / "model.pt").exists():
                models[directory.name] = ("scratch", directory)
    models["rules"] = ("rules", None)
    models["majority"] = ("majority", None)
    return models


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(kind: str, directory: Path | None) -> GapPredictor:
    if kind == "encoder":
        from newlinefix.models.encoder import EncoderGapPredictor

        assert directory is not None
        return EncoderGapPredictor.load(directory)
    if kind == "scratch":
        from newlinefix.models.scratch import ScratchGapPredictor

        assert directory is not None
        return ScratchGapPredictor.load(directory)
    if kind == "rules":
        from newlinefix.models.baseline import RuleBaseline

        return RuleBaseline()
    from newlinefix.models.baseline import AllSpaceBaseline

    return AllSpaceBaseline()


def fix_via_api(text: str, url: str) -> str:
    """Fix text through the served API instead of an in-process predictor."""
    response = httpx.post(f"{url.rstrip('/')}/fix", json={"text": text}, timeout=API_TIMEOUT_S)
    response.raise_for_status()
    return str(response.json()["text"])


st.set_page_config(page_title="Newline Fixer", page_icon="📄", layout="wide")
st.title("📄 Newline Fixer")
st.markdown(
    "Paste text with broken newlines (copy-paste damage, PDF extraction, hard wraps) "
    "and the model re-places them: paragraph breaks, bullet/heading line breaks, and "
    "repairs of words split mid-line. The model can only rewrite whitespace — "
    "**your words are never altered**."
)

models = {"api": ("api", None)} if API_URL else discover_models()
with st.sidebar:
    st.header("Model")
    model_name = st.radio(
        "Choose a model", list(models), format_func=lambda n: f"{n}", label_visibility="collapsed"
    )
    kind, artifact_dir = models[model_name]
    if model_name in MODEL_HELP:
        st.caption(MODEL_HELP[model_name])
    elif artifact_dir is not None:
        st.caption(_describe(kind, artifact_dir))
    if API_URL:
        st.caption(f"`{API_URL}`")
    elif "encoder" not in models:
        st.info("`encoder` and/or `scratch` appear here once trained (see report.md).")

text = st.text_area("Broken text", value=README_EXAMPLE, height=220)
if st.button("Fix newlines", type="primary"):
    if API_URL:
        start = time.perf_counter()
        try:
            fixed = fix_via_api(text, API_URL)
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API at `{API_URL}`: {exc}")
            st.stop()
        elapsed_ms = 1000 * (time.perf_counter() - start)
    else:
        predictor = load_predictor(kind, artifact_dir)
        start = time.perf_counter()
        fixed = fix_text(text, predictor)
        elapsed_ms = 1000 * (time.perf_counter() - start)
    n_words = len(fixed.split())
    left, right = st.columns(2)
    with left:
        st.subheader("Input")
        st.code(text, language=None)
    with right:
        st.subheader(f"Fixed by `{model_name}`")
        st.code(fixed, language=None)
    st.caption(f"{n_words} words in {elapsed_ms:.0f} ms")
