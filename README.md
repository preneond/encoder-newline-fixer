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
experiments, and results: **[report.md](report.md)** · **live demo:
[huggingface.co/spaces/preneond/newlinefix](https://huggingface.co/spaces/preneond/newlinefix)**
(the quantized model running in your browser) · interactive walkthrough:
**[solution presentation](https://claude.ai/code/artifact/9bdefd09-0df2-4632-8a4b-ea48c5cd6a5c)**
(also in this repo: [docs/presentation.html](docs/presentation.html)).

## Quickstart

```bash
uv sync                                        # environment (Python 3.14, uv)
uv run poe test                                # tests
uv run poe serve                               # HTTP API on :8000 (--model picks the checkpoint,
                                               #   --list-models shows what's servable)
uv run poe streamlit                           # interactive UI (uses artifacts/)
```

The API has one endpoint: `POST /fix` with `{"text": "..."}` returns the same words
with fixed whitespace.

### Publishing to the Hugging Face Hub, step by step

A published repo id works everywhere a local artifact dir does —
`EncoderGapPredictor.load`, `evaluate.py --encoder-dir/--extra`, and the API.
Published model: **[preneond/newlinefix-encoder](https://huggingface.co/preneond/newlinefix-encoder)**.

1. **Get a token** — on [huggingface.co](https://huggingface.co) → Settings →
   Access Tokens → create a token with **write** permission.
2. **Authenticate** (either way):

   ```bash
   uv run hf auth login                # paste the token once; stored in ~/.cache/huggingface
   # or, non-interactive (CI, docker):
   export HF_TOKEN=hf_...
   ```

3. **Have a trained model** in `artifacts/encoder` (train it, or check the artifact
   dir contains `model.safetensors` + `predictor_config.json`).
4. **Publish** — creates the repo (private by default), uploads the artifact, and
   generates a model card with the validation metrics:

   ```bash
   uv run poe publish-model --repo-id preneond/newlinefix-encoder
   uv run poe publish-model --repo-id preneond/newlinefix-encoder --no-private   # public instead
   ```

5. **Verify** — the command prints the repo URL; the model card should show the
   metrics table. Then serve straight from the Hub, no local model files needed:

   ```bash
   uv run poe serve --model preneond/newlinefix-encoder
   curl -X POST localhost:8000/fix -H 'Content-Type: application/json' -d '{"text": "que\nries"}'
   ```

   (In Docker/compose the same choice is made with the `NEWLINEFIX_MODEL_DIR` env
   var — the CLI flag is a front-end that sets it.)

Re-running `poe publish-model` pushes a new revision to the same repo (uploads are
commits on the Hub, so the model is versioned for free). The generated corpus
can be published the same way: `uv run poe publish-dataset` uploads
`data/docs/{train,val,test}.jsonl` with a dataset card (private by default —
check the source-corpora licenses before making it public).

Docker (CPU-only torch; the API downloads the published model on first boot,
or serve a local checkpoint by mounting artifacts/ and setting NEWLINEFIX_MODEL_DIR):

```bash
docker build -t newlinefix . && docker run -p 8000:8000 newlinefix
curl -X POST localhost:8000/fix -H 'Content-Type: application/json' -d '{"text": "que\nries"}'

docker compose up                              # API on :8000 + UI on :8501
```

The UI has two modes: under compose it is a thin client of the API (`NEWLINEFIX_API_URL`
is set, so it offers the single served model), while locally it loads predictors
in-process from `artifacts/` and lets you compare all models side by side.

Reproducing the full pipeline (data → training → evaluation) is documented in
[report.md → How to run](report.md#how-to-run).

## Repository layout

| Path | What's there |
|---|---|
| `src/newlinefix/` | Library: gap framing (`gaps.py`), corpus streaming/cleaning (`corpus.py`), self-supervised corruption (`corruption.py`), windowed prediction (`predict.py`), metrics (`metrics.py`), distillation (`distill.py`), HTTP service (`api.py`), models (`models/`) |
| `scripts/` | CLIs: `prepare_data.py`, `train_encoder.py`, `train_scratch.py`, `evaluate.py`, `serve.py`, `publish_model.py`, `publish_dataset.py`, `publish_space.py` |
| `tests/` | 131 unit/property/service tests (`uv run pytest`) |
| `ui/` | Streamlit UI (local model picker / thin API client under compose) |
| `space/` | Static HF Space: in-browser inference via ONNX + transformers.js |
| `artifacts/` | Trained checkpoints (gitignored; train locally, or serving falls back to the published Hub model) |
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
uv run poe serve       # HTTP API on :8000
uv run poe streamlit   # local UI
uv run poe publish-model     # push artifacts/encoder to the HF Hub
uv run poe publish-dataset   # push the generated corpus (private by default)
```

CI (`.github/workflows/ci.yml`) runs `poe check` on every push.
