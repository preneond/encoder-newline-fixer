"""Authoritative evaluation harness comparing all newline-fixing models.

Protocol per document index i (deterministic given --seed):
    clean = normalize(text)
    rng = random.Random(seed * 100_003 + i)
    words, true_labels = make_example(text_to_gaps(clean), rng, CorruptionConfig())
    input_text = render_corrupted(words, true_labels, rng, CorruptionConfig())

Each model predicts gap labels for the input word sequence; predictions are
compared to true_labels directly (aligned by construction). Reconstructed text
is NEVER re-parsed for comparison, because a JOIN prediction merges words and
would break alignment.
"""

import json
import math
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from newlinefix.corruption import CorruptionConfig, make_example, render_corrupted
from newlinefix.data import read_documents
from newlinefix.gaps import (
    JOIN,
    NEWLINE,
    PARA,
    GapText,
    gaps_to_text,
    normalize,
    text_to_gaps,
)
from newlinefix.metrics import (
    accuracy,
    break_prf,
    confusion_matrix,
    edit_similarity,
    macro_f1,
    per_class_prf,
    pk,
    windowdiff,
)
from newlinefix.models.baseline import AllSpaceBaseline, RuleBaseline
from newlinefix.models.encoder import EncoderGapPredictor
from newlinefix.models.scratch import ScratchGapPredictor
from newlinefix.predict import GapPredictor, fix_text

# The broken input example from README.md, verbatim (first ``` fenced block).
README_EXAMPLE = (
    "3.2.3 Applications of Attention\n"
    " in our Model The Transformer uses multi-head attention in three different ways:"
    ' • In "encoder-decoder attention" layers,\n'
    " the que\n"
    "ries come from the previous decoder layer.[...]"
)

WARMUP_WORDS = ["Warm", "up", "call", "so", "lazy", "init", "is", "not", "timed."]

# A doc example is (clean_text, input_words, true_labels).
Example = tuple[str, list[str], list[int]]

console = Console()
console_err = Console(stderr=True)


class EvalConfig(BaseModel):
    """One evaluation run, as parsed from the CLI."""

    data: str
    models: str
    encoder_dir: str
    scratch_dir: str
    limit: int
    seed: int
    out: str


#: --extra KIND values mapped to the loader for that kind of artifact dir.
LOADERS: dict[str, Callable[[str], GapPredictor]] = {
    "encoder": EncoderGapPredictor.load,
    "scratch": ScratchGapPredictor.load,
}


def load_trained(kind: str, artifact_dir: str) -> GapPredictor:
    """Load a trained predictor, failing clearly when artifacts are missing."""
    if not Path(artifact_dir).exists():
        raise FileNotFoundError(f"artifact directory not found: {artifact_dir}")
    return LOADERS[kind](artifact_dir)


REGISTRY: dict[str, Callable[[EvalConfig], GapPredictor]] = {
    "majority": lambda _cfg: AllSpaceBaseline(),
    "rules": lambda _cfg: RuleBaseline(),
    "encoder": lambda cfg: load_trained("encoder", cfg.encoder_dir),
    "scratch": lambda cfg: load_trained("scratch", cfg.scratch_dir),
}


def register_extra(spec: str) -> str:
    """Register an '--extra NAME=KIND:DIR' model in REGISTRY; returns NAME."""
    name, eq, rest = spec.partition("=")
    kind, colon, directory = rest.partition(":")
    if not (name and eq and colon and directory) or kind not in LOADERS:
        raise SystemExit(
            f"bad --extra spec {spec!r}; expected NAME=KIND:DIR with KIND one of "
            + "|".join(LOADERS)
        )
    REGISTRY[name] = lambda _cfg: load_trained(kind, directory)
    return name


def build_examples(texts: list[str], seed: int, cfg: CorruptionConfig) -> list[Example]:
    examples: list[Example] = []
    for i, text in enumerate(tqdm(texts, desc="corrupting", unit="doc")):
        clean = normalize(text)
        # One rng per document (prime stride): doc i's corruption never depends on how
        # many documents precede it, so per-doc results are stable under --limit.
        rng = random.Random(seed * 100_003 + i)
        words, true_labels = make_example(text_to_gaps(clean), rng, cfg)
        input_text = render_corrupted(words, true_labels, rng, cfg)
        input_words = text_to_gaps(input_text).words
        assert input_words == words, "render_corrupted must preserve the word sequence"
        examples.append((clean, input_words, true_labels))
    return examples


def evaluate_model(name: str, predictor: GapPredictor, examples: list[Example]) -> dict:
    predictor.predict_gaps(WARMUP_WORDS)  # warmup: exclude lazy init from timing
    all_true: list[int] = []
    all_pred: list[int] = []
    pks: list[float] = []
    wds: list[float] = []
    sims: list[float] = []
    exact = 0
    total_time = 0.0
    total_words = 0
    for clean, words, true_labels in tqdm(examples, desc=name, unit="doc"):
        t0 = time.perf_counter()
        pred_labels = predictor.predict_gaps(words)
        total_time += time.perf_counter() - t0
        assert len(pred_labels) == len(true_labels), f"{name}: misaligned predictions"
        total_words += len(words)
        all_true.extend(true_labels)
        all_pred.extend(pred_labels)
        pks.append(pk(true_labels, pred_labels))
        wds.append(windowdiff(true_labels, pred_labels))
        pred_text = gaps_to_text(GapText(words, pred_labels))
        sims.append(edit_similarity(pred_text, clean))
        exact += int(pred_text == clean)
    n = len(examples)
    cm = confusion_matrix(all_true, all_pred)
    # pk/windowdiff are NaN for docs too short to probe; exclude them from means.
    pk_values = [v for v in pks if not math.isnan(v)]
    wd_values = [v for v in wds if not math.isnan(v)]
    return {
        "model": name,
        "n_docs": n,
        "n_gaps": len(all_true),
        "n_docs_segmentation_scored": len(pk_values),
        "gap_accuracy": accuracy(cm),
        "per_class": per_class_prf(cm),
        "macro_f1_join_newline_para": macro_f1(cm, [JOIN, NEWLINE, PARA]),
        "break": break_prf(all_true, all_pred),
        "mean_pk": sum(pk_values) / len(pk_values) if pk_values else float("nan"),
        "mean_windowdiff": sum(wd_values) / len(wd_values) if wd_values else float("nan"),
        "mean_edit_similarity": sum(sims) / n,
        "exact_match_rate": exact / n,
        # max() guards the near-instant baselines against division by zero.
        "words_per_sec": total_words / max(total_time, 1e-9),
        "mean_latency_ms": 1000.0 * total_time / n,
        "confusion_matrix": cm.tolist(),
    }


def result_row(r: dict) -> list[str]:
    return [
        r["model"],
        f"{r['gap_accuracy']:.4f}",
        f"{r['macro_f1_join_newline_para']:.4f}",
        f"{r['break']['f1']:.4f}",
        f"{r['mean_pk']:.4f}",
        f"{r['mean_windowdiff']:.4f}",
        f"{r['mean_edit_similarity']:.4f}",
        f"{r['exact_match_rate']:.3f}",
        f"{r['words_per_sec']:,.0f}",
        f"{r['mean_latency_ms']:.2f}",
    ]


RESULT_COLUMNS = (
    "Model",
    "Gap acc",
    "Macro-F1*",
    "Break-F1",
    "Pk",
    "WinDiff",
    "EditSim",
    "Exact",
    "Words/s",
    "ms/doc",
)


def format_table(results: list[dict]) -> str:
    """Markdown table for eval_results.md."""
    rows = [
        "| " + " | ".join(RESULT_COLUMNS) + " |",
        "|" + "---|" * len(RESULT_COLUMNS),
    ]
    rows += ["| " + " | ".join(result_row(r)) + " |" for r in results]
    return "\n".join(rows)


def rich_table(results: list[dict], meta: dict) -> Table:
    table = Table(title=f"{meta['n_docs']} docs, seed {meta['seed']} — {meta['data']}")
    table.add_column(RESULT_COLUMNS[0])
    for column in RESULT_COLUMNS[1:]:
        table.add_column(column, justify="right")
    for r in results:
        table.add_row(*result_row(r))
    return table


def write_report(
    out_dir: Path, results: list[dict], qualitative: list[tuple[str, str]], meta: dict
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Evaluation results",
        "",
        f"Data: `{meta['data']}` — {meta['n_docs']} docs, seed {meta['seed']}.",
        "",
        format_table(results),
        "",
        "\\* Macro-F1 over {JOIN, NEWLINE, PARA} (SPACE excluded as the trivial majority class).",
        "",
        "## Qualitative example (README)",
        "",
        "Input:",
        "",
        "```",
        README_EXAMPLE,
        "```",
        "",
    ]
    for name, fixed in qualitative:
        lines += [f"### {name}", "", "```", fixed, "```", ""]
    (out_dir / "eval_results.md").write_text("\n".join(lines), encoding="utf-8")


@click.command(help=__doc__)
@click.option(
    "--data", default="data/docs/test.jsonl", show_default=True, help="test documents JSONL"
)
@click.option(
    "--models",
    default="all",
    show_default=True,
    help="'all' or comma-list of: majority,rules,encoder,scratch (plus any --extra names)",
)
@click.option("--encoder-dir", default="artifacts/encoder", show_default=True)
@click.option("--scratch-dir", default="artifacts/scratch", show_default=True)
@click.option(
    "--extra",
    multiple=True,
    metavar="NAME=KIND:DIR",
    help="register an additional trained model from an artifact dir, e.g. "
    "electra=encoder:artifacts/encoder-electra-small (repeatable)",
)
@click.option("--limit", type=int, default=500, show_default=True, help="max documents to evaluate")
@click.option("--seed", type=int, default=13, show_default=True)
@click.option("--out", default="results", show_default=True, help="output directory for reports")
def main(extra: tuple[str, ...], **kwargs: Any) -> None:
    cfg = EvalConfig(**kwargs)
    for spec in extra:
        register_extra(spec)
    names = list(REGISTRY) if cfg.models == "all" else [s.strip() for s in cfg.models.split(",")]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        console_err.print(
            f"[red]error:[/red] unknown models {unknown}; choose from {list(REGISTRY)}"
        )
        sys.exit(2)

    texts = list(read_documents(cfg.data))
    if not texts:
        console_err.print(f"[red]error:[/red] no documents found in {cfg.data}")
        sys.exit(2)
    if cfg.limit < len(texts):
        # Splits are written source-grouped; a prefix would evaluate one source only.
        texts = random.Random(cfg.seed).sample(texts, cfg.limit)
    examples = build_examples(texts, cfg.seed, CorruptionConfig())

    results: list[dict] = []
    qualitative: list[tuple[str, str]] = []
    for name in names:
        try:
            predictor = REGISTRY[name](cfg)
        except Exception as exc:  # a missing artifact dir must never block the other models
            console_err.print(f"[yellow]warning:[/yellow] skipping model '{name}': {exc}")
            continue
        results.append(evaluate_model(name, predictor, examples))
        qualitative.append((name, fix_text(README_EXAMPLE, predictor)))

    if not results:
        console_err.print("[red]error:[/red] no models could be evaluated")
        sys.exit(1)

    meta = {"data": cfg.data, "seed": cfg.seed, "limit": cfg.limit, "n_docs": len(examples)}
    write_report(Path(cfg.out), results, qualitative, meta)
    console.print(rich_table(results, meta))
    console.print(
        f"wrote [bold]{cfg.out}/eval_results.json[/] and [bold]{cfg.out}/eval_results.md[/]"
    )


if __name__ == "__main__":
    main()
