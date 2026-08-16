import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    is_script_mode = mo.app_meta().mode == "script"
    return is_script_mode, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Fixing newlines: model development walkthrough

    This notebook drives the full pipeline of the newline-fixing service —
    **data preparation → training → evaluation** — and visualizes the results.

    The task is framed as **gap classification**: the text is split into words, and a
    model predicts one of four separators between each consecutive pair —
    `JOIN` (halves of one word), `SPACE`, `NEWLINE` (`\n`), `PARA` (`\n\n`).
    Reconstruction from words + gaps can only change whitespace, never words.

    Each heavy step is gated behind a **run button** and shells out to the same
    `scripts/*.py` CLIs documented in `report.md`, so what you see here is exactly
    what a terminal run does. Cells render from artifacts on disk, so a notebook
    restart picks up wherever the pipeline left off.
    """)
    return


@app.cell
def _(mo):
    import json
    import subprocess
    import sys
    from html import escape
    from pathlib import Path

    ROOT = mo.notebook_dir().parent
    DATA_DIR = ROOT / "data" / "docs"
    ENCODER_DIR = ROOT / "artifacts" / "encoder"
    SCRATCH_DIR = ROOT / "artifacts" / "scratch"
    RESULTS_DIR = ROOT / "results"

    # Okabe-Ito subset, CVD-validated (dataviz six checks) — fixed model → color mapping.
    MODEL_COLORS = {
        "majority": "#CC79A7",
        "rules": "#E69F00",
        "scratch": "#56B4E9",
        "encoder": "#009E73",
    }
    MODEL_ORDER = ["majority", "rules", "scratch", "encoder"]

    def run_script(args: list[str]) -> tuple[int, str]:
        """Run a repo script with the notebook's interpreter; capture combined output."""
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()

    def pre_block(text: str, max_lines: int = 40):
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = ["… (truncated) …", *lines[-max_lines:]]
        return mo.Html(
            "<pre style='white-space:pre-wrap;font-size:12px;line-height:1.5;"
            "border:1px solid var(--slate-4);border-radius:6px;padding:10px;'>"
            + escape("\n".join(lines))
            + "</pre>"
        )

    def read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    return (
        DATA_DIR,
        ENCODER_DIR,
        MODEL_COLORS,
        MODEL_ORDER,
        RESULTS_DIR,
        SCRATCH_DIR,
        json,
        pre_block,
        read_json,
        run_script,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Data preparation

    Clean structured text is its own supervision: we record the true gap classes,
    then destroy the newlines (and occasionally split words) to create inputs.
    Sources: **wikitext-103** (prose + headings) and a **markdown corpus**
    (headings + bullet lists, like the README's arXiv example).

    Small doc counts below give a quick demo corpus; raise them for the real run
    (the full run is also fine from a terminal — the button uses the same CLI).
    """)
    return


@app.cell
def _(mo):
    wikitext_n = mo.ui.number(value=12000, start=10, stop=100_000, step=10, label="wikitext docs")
    markdown_n = mo.ui.number(value=12000, start=0, stop=100_000, step=10, label="markdown docs")
    prep_btn = mo.ui.run_button(label="Run data preparation")
    mo.hstack([wikitext_n, markdown_n, prep_btn], justify="start", gap=2)
    return markdown_n, prep_btn, wikitext_n


@app.cell
def _(DATA_DIR, markdown_n, mo, pre_block, prep_btn, run_script, wikitext_n):
    _log = None
    if prep_btn.value:
        _code, _out = run_script(
            [
                "scripts/prepare_data.py",
                "--out",
                str(DATA_DIR),
                "--wikitext-docs",
                str(int(wikitext_n.value)),
                "--markdown-docs",
                str(int(markdown_n.value)),
            ]
        )
        _log = f"$ prepare_data.py → exit {_code}\n{_out}"
    prep_done = (DATA_DIR / "train.jsonl").exists()

    if _log is not None:
        _display = pre_block(_log)
    elif prep_done:
        _display = mo.md(f"✅ Prepared corpus found at `{DATA_DIR}` — cells below use it.")
    else:
        _display = mo.md("⚠️ _No prepared corpus yet — click **Run data preparation** above._")
    _display
    return (prep_done,)


@app.cell
def _(DATA_DIR, is_script_mode, json, prep_done):
    from collections import Counter

    from newlinefix.gaps import NEWLINE, PARA, SPACE, text_to_gaps

    def _split_rows(split: str, limit: int = 4000) -> list[dict]:
        rows = []
        with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as _f:
            for _i, _line in enumerate(_f):
                if _i >= limit:
                    break
                _doc = json.loads(_line)
                _gaps = Counter(text_to_gaps(_doc["text"]).gaps)
                rows.append(
                    {
                        "split": split,
                        "source": _doc["source"],
                        "words": _gaps.total() + 1,
                        "SPACE": _gaps[SPACE],
                        "NEWLINE": _gaps[NEWLINE],
                        "PARA": _gaps[PARA],
                    }
                )
        return rows

    corpus_rows = (
        [_row for _s in ("train", "val", "test") for _row in _split_rows(_s)] if prep_done else []
    )
    if is_script_mode and corpus_rows:
        print(f"corpus sample: {len(corpus_rows)} docs across splits")
    return (corpus_rows,)


@app.cell
def _(corpus_rows, is_script_mode, mo):
    import pandas as pd

    corpus_df = pd.DataFrame(corpus_rows)
    if corpus_df.empty:
        _display = mo.md("_Corpus stats appear here after data preparation._")
    else:
        _summary = (
            corpus_df.groupby(["split", "source"])
            .agg(
                docs=("words", "size"),
                words=("words", "sum"),
                newline_gaps=("NEWLINE", "sum"),
                para_gaps=("PARA", "sum"),
            )
            .reset_index()
        )
        if is_script_mode:
            print(_summary.to_string(index=False))
        _display = mo.vstack(
            [
                mo.md("**Corpus composition** (sampled up to 4000 docs/split):"),
                mo.ui.table(_summary),
            ]
        )
    _display
    return (corpus_df,)


@app.cell
def _(MODEL_COLORS, corpus_df, mo):
    import plotly.express as px

    mo.stop(corpus_df.empty, mo.md(""))
    _agg = corpus_df.groupby(["split", "source"]).size().reset_index(name="docs")
    fig_docs = px.bar(
        _agg,
        x="split",
        y="docs",
        color="source",
        barmode="group",
        title="Documents per split and source",
        template="simple_white",
        color_discrete_sequence=[MODEL_COLORS["encoder"], MODEL_COLORS["scratch"]],
        category_orders={"split": ["train", "val", "test"]},
        text_auto=True,
        height=380,
    )
    fig_docs.update_traces(textposition="outside")
    mo.ui.plotly(fig_docs)
    return (px,)


@app.cell
def _(MODEL_COLORS, corpus_df, mo, px):
    mo.stop(corpus_df.empty, mo.md(""))
    _gap_share = corpus_df.groupby("source")[["SPACE", "NEWLINE", "PARA"]].sum().reset_index()
    for _cls in ("NEWLINE", "PARA"):
        _gap_share[_cls + " %"] = (
            100 * _gap_share[_cls] / (_gap_share[["SPACE", "NEWLINE", "PARA"]].sum(axis=1))
        )
    _melted = _gap_share.melt(
        id_vars="source",
        value_vars=["NEWLINE %", "PARA %"],
        var_name="gap class",
        value_name="share",
    )
    fig_breaks = px.bar(
        _melted,
        x="source",
        y="share",
        color="gap class",
        barmode="group",
        title="Break-gap share by source (% of all gaps that are line/paragraph breaks)",
        template="simple_white",
        height=380,
        text_auto=".2f",
        color_discrete_sequence=[MODEL_COLORS["rules"], MODEL_COLORS["majority"]],
        labels={"share": "% of gaps"},
    )
    fig_breaks.update_traces(textposition="outside")
    mo.ui.plotly(fig_breaks)
    return


@app.cell
def _(MODEL_COLORS, corpus_df, mo, px):
    mo.stop(corpus_df.empty, mo.md(""))
    fig_len = px.histogram(
        corpus_df,
        x="words",
        color="source",
        barmode="overlay",
        opacity=0.65,
        nbins=60,
        title="Document length distribution (words)",
        template="simple_white",
        height=380,
        color_discrete_sequence=[MODEL_COLORS["encoder"], MODEL_COLORS["scratch"]],
    )
    mo.ui.plotly(fig_len)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Corruption: how training pairs are made

    Pick a document; left is the **clean target**, right is the **corrupted input**
    the model sees (newlines destroyed, occasional spurious breaks and mid-word
    splits). The true gap labels ride along, aligned to the corrupted words.
    """)
    return


@app.cell
def _(mo):
    doc_idx = mo.ui.slider(0, 200, value=0, label="validation doc #", show_value=True)
    doc_idx
    return (doc_idx,)


@app.cell
def _(DATA_DIR, doc_idx, mo, pre_block, prep_done):
    import itertools
    import random

    from newlinefix.corruption import CorruptionConfig, make_example, render_corrupted
    from newlinefix.data import read_documents
    from newlinefix.gaps import text_to_gaps as _t2g

    mo.stop(not prep_done, mo.md("_Prepare data first to see a corruption example._"))
    _doc = next(itertools.islice(read_documents(DATA_DIR / "val.jsonl"), doc_idx.value, None), None)
    mo.stop(_doc is None, mo.md("_Doc index out of range for this corpus size._"))
    _doc = " ".join(_doc.split()[:180]) if len(_doc.split()) > 180 else _doc
    _rng = random.Random(doc_idx.value)
    _cfg = CorruptionConfig()
    _words, _labels = make_example(_t2g(_doc), _rng, _cfg)
    _broken = render_corrupted(_words, _labels, _rng, _cfg)
    mo.hstack(
        [
            mo.vstack([mo.md("**Clean target**"), pre_block(_doc)]),
            mo.vstack([mo.md("**Corrupted input**"), pre_block(_broken)]),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Training

    Two models learn the same gap-classification task:

    | | `encoder` (served model) | `scratch` (comparison) |
    |---|---|---|
    | Architecture | `distilroberta-base` token classifier | byte-level BiLSTM (~3M params) |
    | Pretraining | yes (82M params, fine-tuned) | none — trained only on our data |

    Defaults below are a **quick demo run**; for the real training raise
    epochs/windows (or run the CLI from a terminal — same script, same flags).
    Both scripts pick MPS automatically on Apple Silicon.
    """)
    return


@app.cell
def _(mo):
    enc_epochs = mo.ui.number(value=1, start=1, stop=10, label="encoder epochs")
    enc_windows = mo.ui.number(
        value=5000, start=100, stop=2_000_000, step=100, label="encoder train windows"
    )
    train_enc_btn = mo.ui.run_button(label="Train encoder")
    scr_epochs = mo.ui.number(value=1, start=1, stop=20, label="scratch epochs")
    scr_windows = mo.ui.number(
        value=5000, start=100, stop=2_000_000, step=100, label="scratch train windows"
    )
    train_scr_btn = mo.ui.run_button(label="Train scratch")
    mo.vstack(
        [
            mo.hstack([enc_epochs, enc_windows, train_enc_btn], justify="start", gap=2),
            mo.hstack([scr_epochs, scr_windows, train_scr_btn], justify="start", gap=2),
        ]
    )
    return (
        enc_epochs,
        enc_windows,
        scr_epochs,
        scr_windows,
        train_enc_btn,
        train_scr_btn,
    )


@app.cell
def _(
    DATA_DIR,
    ENCODER_DIR,
    enc_epochs,
    enc_windows,
    mo,
    pre_block,
    run_script,
    train_enc_btn,
):
    _log = None
    if train_enc_btn.value:
        _code, _out = run_script(
            [
                "scripts/train_encoder.py",
                "--data",
                str(DATA_DIR),
                "--out",
                str(ENCODER_DIR),
                "--epochs",
                str(int(enc_epochs.value)),
                "--train-windows",
                str(int(enc_windows.value)),
            ]
        )
        _log = f"$ train_encoder.py → exit {_code}\n{_out}"
    encoder_ready = (ENCODER_DIR / "predictor_config.json").exists()
    _display = (
        pre_block(_log)
        if _log
        else mo.md(
            f"✅ Encoder artifacts found at `{ENCODER_DIR}`."
            if encoder_ready
            else "⚠️ _No trained encoder yet._"
        )
    )
    _display
    return (encoder_ready,)


@app.cell
def _(
    DATA_DIR,
    SCRATCH_DIR,
    mo,
    pre_block,
    run_script,
    scr_epochs,
    scr_windows,
    train_scr_btn,
):
    _log = None
    if train_scr_btn.value:
        _code, _out = run_script(
            [
                "scripts/train_scratch.py",
                "--data",
                str(DATA_DIR),
                "--out",
                str(SCRATCH_DIR),
                "--epochs",
                str(int(scr_epochs.value)),
                "--train-windows",
                str(int(scr_windows.value)),
            ]
        )
        _log = f"$ train_scratch.py → exit {_code}\n{_out}"
    scratch_ready = (SCRATCH_DIR / "config.json").exists()
    _display = (
        pre_block(_log)
        if _log
        else mo.md(
            f"✅ Scratch artifacts found at `{SCRATCH_DIR}`."
            if scratch_ready
            else "⚠️ _No trained scratch model yet._"
        )
    )
    _display
    return (scratch_ready,)


@app.cell
def _(
    ENCODER_DIR,
    SCRATCH_DIR,
    encoder_ready,
    is_script_mode,
    read_json,
    scratch_ready,
):
    import pandas as _pd

    def _log_frames(name: str, path) -> tuple:
        _log = read_json(path / "training_log.json")
        if not _log:
            return None, None
        _steps = _pd.DataFrame(_log.get("steps", []))
        _epochs = _pd.DataFrame(_log.get("epochs", []))
        for _df in (_steps, _epochs):
            if not _df.empty:
                _df["model"] = name
        return _steps, _epochs

    _enc_steps, _enc_epochs = _log_frames("encoder", ENCODER_DIR) if encoder_ready else (None, None)
    _scr_steps, _scr_epochs = _log_frames("scratch", SCRATCH_DIR) if scratch_ready else (None, None)
    steps_df = (
        _pd.concat([_df for _df in (_enc_steps, _scr_steps) if _df is not None])
        if (_enc_steps is not None or _scr_steps is not None)
        else _pd.DataFrame()
    )
    epochs_df = (
        _pd.concat([_df for _df in (_enc_epochs, _scr_epochs) if _df is not None])
        if (_enc_epochs is not None or _scr_epochs is not None)
        else _pd.DataFrame()
    )
    if is_script_mode and not epochs_df.empty:
        print(epochs_df.to_string(index=False))
    return epochs_df, steps_df


@app.cell
def _(MODEL_COLORS, mo, px, steps_df):
    mo.stop(
        steps_df.empty,
        mo.md(
            "_Training loss curves appear here once a model has been trained (training_log.json)._"
        ),
    )
    fig_loss = px.line(
        steps_df,
        x="step",
        y="loss",
        color="model",
        title="Training loss (weighted cross-entropy)",
        template="simple_white",
        height=400,
        color_discrete_map=MODEL_COLORS,
    )
    mo.ui.plotly(fig_loss)
    return


@app.cell
def _(MODEL_COLORS, epochs_df, mo, px):
    mo.stop(epochs_df.empty or "macro_f1_structural" not in epochs_df.columns, mo.md(""))
    fig_val = px.line(
        epochs_df,
        x="epoch",
        y="macro_f1_structural",
        color="model",
        markers=True,
        title="Validation macro-F1 over JOIN/NEWLINE/PARA per epoch",
        template="simple_white",
        height=400,
        color_discrete_map=MODEL_COLORS,
    )
    fig_val.update_yaxes(range=[0, 1])
    mo.ui.plotly(fig_val)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Evaluation

    Held-out test docs are corrupted with a fixed seed, every model re-predicts all
    gaps, and we score: per-class F1, break detection, segmentation quality
    (Pk / WindowDiff, lower = better), text similarity, and throughput.
    Baselines (`majority`, `rules`) show what the learned models actually buy.
    """)
    return


@app.cell
def _(mo):
    eval_limit = mo.ui.number(value=300, start=10, stop=5000, step=10, label="test docs")
    eval_btn = mo.ui.run_button(label="Run evaluation")
    mo.hstack([eval_limit, eval_btn], justify="start", gap=2)
    return eval_btn, eval_limit


@app.cell
def _(DATA_DIR, RESULTS_DIR, eval_btn, eval_limit, mo, pre_block, run_script):
    _log = None
    if eval_btn.value:
        _code, _out = run_script(
            [
                "scripts/evaluate.py",
                "--data",
                str(DATA_DIR / "test.jsonl"),
                "--models",
                "all",
                "--limit",
                str(int(eval_limit.value)),
                "--out",
                str(RESULTS_DIR),
            ]
        )
        _log = f"$ evaluate.py → exit {_code}\n{_out}"
    eval_ready = (RESULTS_DIR / "eval_results.json").exists()
    _display = (
        pre_block(_log)
        if _log
        else mo.md(
            f"✅ Evaluation results found at `{RESULTS_DIR}/eval_results.json`."
            if eval_ready
            else "⚠️ _No evaluation results yet._"
        )
    )
    _display
    return (eval_ready,)


@app.cell
def _(RESULTS_DIR, eval_ready, read_json):
    _raw = read_json(RESULTS_DIR / "eval_results.json") if eval_ready else None
    # Adapt scripts/evaluate.py's schema ({"results": [...]}) to the notebook's flat view.
    model_metrics: dict = {
        _r["model"]: {
            "gap_accuracy": _r["gap_accuracy"],
            "per_class": _r["per_class"],
            "macro_f1_breaks": _r["macro_f1_join_newline_para"],
            "break": _r["break"],
            "pk": _r["mean_pk"],
            "windowdiff": _r["mean_windowdiff"],
            "edit_similarity": _r["mean_edit_similarity"],
            "exact_match": _r["exact_match_rate"],
            "words_per_sec": _r["words_per_sec"],
            "confusion": _r["confusion_matrix"],
        }
        for _r in (_raw or {}).get("results", [])
    }
    return (model_metrics,)


@app.cell
def _(MODEL_ORDER, is_script_mode, mo, model_metrics: dict):
    import pandas as __pd

    mo.stop(not model_metrics, mo.md("_Run the evaluation to see the comparison table._"))
    _rows = []
    for _name in [m for m in MODEL_ORDER if m in model_metrics] + [
        m for m in model_metrics if m not in MODEL_ORDER
    ]:
        _m = model_metrics[_name]
        _rows.append(
            {
                "model": _name,
                "gap acc": round(_m.get("gap_accuracy", float("nan")), 4),
                "macro-F1 (JOIN/NL/PARA)": round(_m.get("macro_f1_breaks", float("nan")), 4),
                "break F1": round(_m.get("break", {}).get("f1", float("nan")), 4),
                "Pk ↓": round(_m.get("pk", float("nan")), 4),
                "WinDiff ↓": round(_m.get("windowdiff", float("nan")), 4),
                "edit sim": round(_m.get("edit_similarity", float("nan")), 4),
                "exact %": round(100 * _m.get("exact_match", float("nan")), 2),
                "words/s": int(_m.get("words_per_sec", 0)),
            }
        )
    metrics_table = __pd.DataFrame(_rows)
    if is_script_mode:
        print(metrics_table.to_string(index=False))
    mo.ui.table(metrics_table, selection=None)
    return


@app.cell
def _(MODEL_COLORS, MODEL_ORDER, mo, model_metrics: dict, px):
    import pandas as ___pd

    mo.stop(not model_metrics, mo.md(""))
    _rows = [
        {"model": _name, "gap class": _cls, "F1": _stats.get("f1", 0.0)}
        for _name in model_metrics
        for _cls, _stats in model_metrics[_name].get("per_class", {}).items()
    ]
    fig_f1 = px.bar(
        ___pd.DataFrame(_rows),
        x="gap class",
        y="F1",
        color="model",
        barmode="group",
        title="Per-class F1 by model",
        template="simple_white",
        height=420,
        color_discrete_map=MODEL_COLORS,
        text_auto=".2f",
        category_orders={
            "gap class": ["JOIN", "SPACE", "NEWLINE", "PARA"],
            "model": MODEL_ORDER,
        },
    )
    fig_f1.update_traces(textposition="outside")
    fig_f1.update_yaxes(range=[0, 1.05])
    mo.ui.plotly(fig_f1)
    return


@app.cell
def _(MODEL_COLORS, MODEL_ORDER, mo, model_metrics: dict, px):
    import pandas as ____pd

    mo.stop(not model_metrics, mo.md(""))
    _rows = [
        {
            "model": _name,
            "words/s": _m.get("words_per_sec", float("nan")),
            "macro-F1": _m.get("macro_f1_breaks", float("nan")),
        }
        for _name, _m in model_metrics.items()
    ]
    fig_qt = px.scatter(
        ____pd.DataFrame(_rows),
        x="words/s",
        y="macro-F1",
        color="model",
        text="model",
        log_x=True,
        title="Quality vs throughput (up and to the right is better)",
        template="simple_white",
        height=420,
        color_discrete_map=MODEL_COLORS,
        category_orders={"model": MODEL_ORDER},
    )
    fig_qt.update_traces(textposition="top center", marker={"size": 14})
    fig_qt.update_yaxes(range=[0, 1.05])
    mo.ui.plotly(fig_qt)
    return


@app.cell
def _(mo, model_metrics: dict):
    _available = (
        [_n for _n, _m in model_metrics.items() if "confusion" in _m] if model_metrics else []
    )
    cm_model_dd = (
        mo.ui.dropdown(_available, value=_available[-1], label="confusion matrix for")
        if _available
        else None
    )
    cm_model_dd if cm_model_dd is not None else mo.md("")
    return (cm_model_dd,)


@app.cell
def _(cm_model_dd, mo, model_metrics: dict, px):
    import numpy as _np

    mo.stop(cm_model_dd is None, mo.md(""))
    _labels = ["JOIN", "SPACE", "NEWLINE", "PARA"]
    _cm = _np.asarray(model_metrics[cm_model_dd.value]["confusion"], dtype=float)
    _row_sums = _cm.sum(axis=1, keepdims=True)
    _norm = _np.divide(_cm, _row_sums, out=_np.zeros_like(_cm), where=_row_sums > 0)
    fig_cm = px.imshow(
        _norm,
        x=_labels,
        y=_labels,
        text_auto=".3f",
        color_continuous_scale="Blues",
        title=f"Row-normalized confusion — {cm_model_dd.value} (rows = true class)",
        template="simple_white",
        height=440,
        zmin=0,
        zmax=1,
        labels={"x": "predicted", "y": "true"},
    )
    mo.ui.plotly(fig_cm)
    return


@app.cell
def _(MODEL_COLORS, MODEL_ORDER, mo, model_metrics: dict, px):
    import pandas as _____pd

    mo.stop(not model_metrics, mo.md(""))
    _rows = [
        {"model": _name, "metric": _metric, "value": _m.get(_key, float("nan"))}
        for _name, _m in model_metrics.items()
        for _metric, _key in (("Pk", "pk"), ("WindowDiff", "windowdiff"))
    ]
    fig_seg = px.bar(
        _____pd.DataFrame(_rows),
        x="metric",
        y="value",
        color="model",
        barmode="group",
        title="Paragraph segmentation error (lower is better)",
        template="simple_white",
        height=380,
        color_discrete_map=MODEL_COLORS,
        text_auto=".3f",
        category_orders={"model": MODEL_ORDER},
    )
    fig_seg.update_traces(textposition="outside")
    mo.ui.plotly(fig_seg)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 · Try it

    The exact broken example from the challenge README is pre-filled — paste
    anything else to see a model fix it live.
    """)
    return


@app.cell
def _(mo):
    README_BROKEN = (
        "3.2.3 Applications of Attention\n"
        " in our Model The Transformer uses multi-head attention in three different ways: "
        '• In "encoder-decoder attention" layers,\n'
        " the que\n"
        "ries come from the previous decoder layer.[...]"
    )
    try_text = mo.ui.text_area(value=README_BROKEN, rows=6, label="Broken text", full_width=True)
    try_model_dd = mo.ui.dropdown(
        ["encoder", "scratch", "rules", "majority"], value="encoder", label="model"
    )
    try_btn = mo.ui.run_button(label="Fix newlines")
    mo.vstack([try_text, mo.hstack([try_model_dd, try_btn], justify="start", gap=2)])
    return try_btn, try_model_dd, try_text


@app.cell
def _():
    predictor_cache: dict = {}
    return (predictor_cache,)


@app.cell
def _(
    ENCODER_DIR,
    SCRATCH_DIR,
    mo,
    pre_block,
    predictor_cache: dict,
    try_btn,
    try_model_dd,
    try_text,
):
    from newlinefix.predict import fix_text

    mo.stop(not try_btn.value, mo.md("_Click **Fix newlines** to run the selected model._"))

    def _get_predictor(name: str):
        if name not in predictor_cache:
            if name == "encoder":
                from newlinefix.models.encoder import EncoderGapPredictor

                predictor_cache[name] = EncoderGapPredictor.load(ENCODER_DIR)
            elif name == "scratch":
                from newlinefix.models.scratch import ScratchGapPredictor

                predictor_cache[name] = ScratchGapPredictor.load(SCRATCH_DIR)
            elif name == "rules":
                from newlinefix.models.baseline import RuleBaseline

                predictor_cache[name] = RuleBaseline()
            else:
                from newlinefix.models.baseline import AllSpaceBaseline

                predictor_cache[name] = AllSpaceBaseline()
        return predictor_cache[name]

    try:
        _fixed = fix_text(try_text.value, _get_predictor(try_model_dd.value))
        _result = mo.hstack(
            [
                mo.vstack([mo.md("**Input**"), pre_block(try_text.value)]),
                mo.vstack([mo.md(f"**Fixed by `{try_model_dd.value}`**"), pre_block(_fixed)]),
            ],
            widths="equal",
            gap=1,
        )
    except (FileNotFoundError, OSError, ImportError) as _err:
        _result = mo.md(f"⚠️ _Could not load `{try_model_dd.value}`: {_err}_")
    _result
    return


if __name__ == "__main__":
    app.run()
