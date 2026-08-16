"""Minimal Streamlit UI for the newline-fixing models.

Run locally:  uv run streamlit run ui/streamlit_app.py
Models are detected from their artifact directories; baselines are always available.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from newlinefix.predict import GapPredictor, fix_text

ROOT = Path(__file__).resolve().parent.parent
ENCODER_DIR = ROOT / "artifacts" / "encoder"
SCRATCH_DIR = ROOT / "artifacts" / "scratch"

README_EXAMPLE = (
    "3.2.3 Applications of Attention\n"
    " in our Model The Transformer uses multi-head attention in three different ways:"
    ' • In "encoder-decoder attention" layers,\n'
    " the que\n"
    "ries come from the previous decoder layer.[...]"
)

MODEL_HELP = {
    "encoder": "fine-tuned distilroberta-base token-gap classifier (the served model)",
    "scratch": "byte-level BiLSTM (~2.4M params) trained from scratch on our data",
    "rules": "hand-written rules: newline before bullets, break before numbered headings",
    "majority": "always predicts a plain space (the trivial baseline)",
}


def available_models() -> list[str]:
    models = []
    if (ENCODER_DIR / "predictor_config.json").exists():
        models.append("encoder")
    if (SCRATCH_DIR / "config.json").exists():
        models.append("scratch")
    return [*models, "rules", "majority"]


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(name: str) -> GapPredictor:
    if name == "encoder":
        from newlinefix.models.encoder import EncoderGapPredictor

        return EncoderGapPredictor.load(ENCODER_DIR)
    if name == "scratch":
        from newlinefix.models.scratch import ScratchGapPredictor

        return ScratchGapPredictor.load(SCRATCH_DIR)
    if name == "rules":
        from newlinefix.models.baseline import RuleBaseline

        return RuleBaseline()
    from newlinefix.models.baseline import AllSpaceBaseline

    return AllSpaceBaseline()


st.set_page_config(page_title="Newline Fixer", page_icon="📄", layout="wide")
st.title("📄 Newline Fixer")
st.markdown(
    "Paste text with broken newlines (copy-paste damage, PDF extraction, hard wraps) "
    "and the model re-places them: paragraph breaks, bullet/heading line breaks, and "
    "repairs of words split mid-line. The model can only rewrite whitespace — "
    "**your words are never altered**."
)

models = available_models()
with st.sidebar:
    st.header("Model")
    model_name = st.radio(
        "Choose a model", models, format_func=lambda n: f"{n}", label_visibility="collapsed"
    )
    st.caption(MODEL_HELP[model_name])
    if "encoder" not in models:
        st.info("`encoder` and/or `scratch` appear here once trained (see report.md).")

text = st.text_area("Broken text", value=README_EXAMPLE, height=220)
if st.button("Fix newlines", type="primary"):
    predictor = load_predictor(model_name)
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
