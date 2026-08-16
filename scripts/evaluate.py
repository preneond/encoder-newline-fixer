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

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

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


def _make_majority(_args: argparse.Namespace) -> GapPredictor:
    from newlinefix.models.baseline import AllSpaceBaseline

    return AllSpaceBaseline()


def _make_rules(_args: argparse.Namespace) -> GapPredictor:
    from newlinefix.models.baseline import RuleBaseline

    return RuleBaseline()


def _load_trained(module_name: str, artifact_dir: str) -> GapPredictor:
    """Load a trained predictor from its module by convention.

    Prefers a module-level ``load_predictor(dir)``; otherwise any GapPredictor
    subclass exposing a callable ``load(dir)``. Raises if nothing loadable is
    found — the caller treats that as "model unavailable, skip".
    """
    if not Path(artifact_dir).exists():
        raise FileNotFoundError(f"artifact directory not found: {artifact_dir}")
    module = importlib.import_module(module_name)
    loader = getattr(module, "load_predictor", None)
    if callable(loader):
        return loader(artifact_dir)
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, GapPredictor)
            and obj is not GapPredictor
            and callable(getattr(obj, "load", None))
        ):
            return obj.load(artifact_dir)  # type: ignore[attr-defined]
    raise RuntimeError(f"no load_predictor() or GapPredictor.load() found in {module_name}")


def _make_encoder(args: argparse.Namespace) -> GapPredictor:
    return _load_trained("newlinefix.models.encoder", args.encoder_dir)


def _make_scratch(args: argparse.Namespace) -> GapPredictor:
    return _load_trained("newlinefix.models.scratch", args.scratch_dir)


REGISTRY: dict[str, Callable[[argparse.Namespace], GapPredictor]] = {
    "majority": _make_majority,
    "rules": _make_rules,
    "encoder": _make_encoder,
    "scratch": _make_scratch,
}


def build_examples(texts: list[str], seed: int, cfg: CorruptionConfig) -> list[Example]:
    examples: list[Example] = []
    for i, text in enumerate(tqdm(texts, desc="corrupting", unit="doc")):
        clean = normalize(text)
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
        "words_per_sec": total_words / max(total_time, 1e-9),
        "mean_latency_ms": 1000.0 * total_time / n,
        "confusion_matrix": cm.tolist(),
    }


def format_table(results: list[dict]) -> str:
    header = (
        "| Model | Gap acc | Macro-F1* | Break-F1 | Pk | WinDiff | EditSim "
        "| Exact | Words/s | ms/doc |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        rows.append(
            f"| {r['model']} | {r['gap_accuracy']:.4f} | {r['macro_f1_join_newline_para']:.4f} "
            f"| {r['break']['f1']:.4f} | {r['mean_pk']:.4f} | {r['mean_windowdiff']:.4f} "
            f"| {r['mean_edit_similarity']:.4f} | {r['exact_match_rate']:.3f} "
            f"| {r['words_per_sec']:,.0f} | {r['mean_latency_ms']:.2f} |"
        )
    return "\n".join(rows)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/docs/test.jsonl", help="test documents JSONL")
    parser.add_argument(
        "--models",
        default="all",
        help="'all' or comma-list of: " + ",".join(REGISTRY),
    )
    parser.add_argument("--encoder-dir", default="artifacts/encoder")
    parser.add_argument("--scratch-dir", default="artifacts/scratch")
    parser.add_argument("--limit", type=int, default=500, help="max documents to evaluate")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="results", help="output directory for reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = list(REGISTRY) if args.models == "all" else [s.strip() for s in args.models.split(",")]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"error: unknown models {unknown}; choose from {list(REGISTRY)}", file=sys.stderr)
        return 2

    texts = list(read_documents(args.data))
    if not texts:
        print(f"error: no documents found in {args.data}", file=sys.stderr)
        return 2
    if args.limit < len(texts):
        # Splits are written source-grouped; a prefix would evaluate one source only.
        texts = random.Random(args.seed).sample(texts, args.limit)
    examples = build_examples(texts, args.seed, CorruptionConfig())

    results: list[dict] = []
    qualitative: list[tuple[str, str]] = []
    for name in names:
        try:
            predictor = REGISTRY[name](args)
        except Exception as exc:  # missing artifacts/deps must never block baselines
            print(f"warning: skipping model '{name}': {exc}", file=sys.stderr)
            continue
        results.append(evaluate_model(name, predictor, examples))
        qualitative.append((name, fix_text(README_EXAMPLE, predictor)))

    if not results:
        print("error: no models could be evaluated", file=sys.stderr)
        return 1

    meta = {"data": args.data, "seed": args.seed, "limit": args.limit, "n_docs": len(examples)}
    write_report(Path(args.out), results, qualitative, meta)
    print(format_table(results))
    print(f"\nwrote {args.out}/eval_results.json and {args.out}/eval_results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
