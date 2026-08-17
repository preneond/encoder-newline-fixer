# Newline Fixer

A machine-learning service that fixes newline placement in English text: paragraph
breaks, line breaks before bullets and headings, and repairs of words split mid-line.
Built as a solution to the [BottleCapAI](https://www.bottlecapai.com) Applied ML
Engineer challenge.

Example — broken input:

```
3.2.3 Applications of Attention
 in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers,
 the que
ries come from the previous decoder layer.[...]
```

fixed output:

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:
• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.
[...]
```

The task is framed as **4-class gap classification** between consecutive words
(`JOIN` / `SPACE` / `NEWLINE` / `PARA`), so the model can only rewrite whitespace —
the output words are guaranteed identical to the input. The served model is a
fine-tuned `distilroberta-base` token classifier; it is compared against a
from-scratch byte-level BiLSTM and two non-neural baselines. Full methodology,
experiments, and results: **[report.md](report.md)** · interactive walkthrough:
**[solution presentation](https://claude.ai/code/artifact/9bdefd09-0df2-4632-8a4b-ea48c5cd6a5c)**
(also in this repo: [docs/presentation.html](docs/presentation.html)).

## Quickstart

```bash
uv sync                                        # environment (Python 3.14, uv)
uv run poe test                                # tests
uv run poe streamlit                           # interactive UI (uses artifacts/)
```

Reproducing the full pipeline (data → training → evaluation) is documented in
[report.md → How to run](report.md#how-to-run).

## Repository layout

| Path | What's there |
|---|---|
| `src/newlinefix/` | Library: gap framing (`gaps.py`), corpus streaming/cleaning (`corpora.py`), self-supervised corruption (`corruption.py`), windowed prediction (`predict.py`), metrics (`metrics.py`), distillation (`distill.py`), models (`models/`) |
| `scripts/` | CLIs: `prepare_data.py` (build corpus), `train_encoder.py` (fine-tune a pretrained encoder), `train_scratch.py` (byte-BiLSTM, optional distillation), `evaluate.py` (compare all models on the held-out test split) |
| `tests/` | 106 unit/property tests (`uv run pytest`) |
| `ui/` | Minimal Streamlit UI: model picker, before/after view, latency readout |
| `artifacts/` | Trained model checkpoints (`encoder` is the served model) |
| `results/` | Evaluation reports: headline `eval_results.{json,md}`, exploration sweeps in `exploration_results.{json,md}` |
| `report.md` | Approach, decisions, experiments, and results |
| `data/` | Generated corpus (gitignored; regenerate with `scripts/prepare_data.py`) |

## Quality gates

```bash
uv run poe check       # lint + typecheck + test (what CI runs)
uv run poe lint        # ruff check
uv run poe fix         # ruff check --fix + ruff format
uv run poe typecheck   # ty check
uv run poe test        # pytest
```

CI (`.github/workflows/ci.yml`) runs `poe check` on every push.
