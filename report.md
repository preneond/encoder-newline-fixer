# Newline Fixer — report

A machine-learning service that fixes newline placement in English text: paragraph
breaks (`\n\n`), line-level breaks (`\n` before bullets and headings), and repairs of
words split mid-line (`que\nries` → `queries`).

## TL;DR

- The task is framed as **4-class gap classification** between consecutive words —
  `JOIN` / `SPACE` / `NEWLINE` / `PARA` — so the model can only rewrite whitespace:
  **the output words are guaranteed identical to the input**, by construction.
- Training data is **self-supervised**: clean structured text (Wikipedia + arXiv
  markdown) is its own label source; inputs are produced by programmatically
  destroying newlines and splitting words.
- The served model is a **fine-tuned `distilroberta-base`** token classifier. It is
  compared against a **from-scratch byte-level BiLSTM** (~2.4M params) and two
  non-neural baselines — see [Results](#results).

## How to run

```bash
# environment (Python 3.12, uv)
uv sync

# tests, lint, types
uv run pytest
uv run ruff check src tests scripts
uv run ty check src tests scripts

# 1. data: stream + clean corpora into canonical documents (train/val/test JSONL)
uv run python scripts/prepare_data.py --out data/docs --wikitext-docs 12000 --markdown-docs 12000

# 2. train the two models (auto-selects MPS/CUDA/CPU)
uv run python scripts/train_encoder.py --data data/docs --out artifacts/encoder --epochs 2 --train-windows 80000
uv run python scripts/train_scratch.py --data data/docs --out artifacts/scratch --epochs 3

# 3. evaluate all models on the held-out test split
uv run python scripts/evaluate.py --models all --out results

# interactive walkthrough (data prep -> training -> eval, with charts)
uv run marimo edit notebooks/model_development.py

# minimal UI
uv run streamlit run ui/streamlit_app.py
```

## Approach

### Task framing: gap classification, not generation

Text is split into whitespace-delimited words; between each consecutive pair the model
predicts one of four separators:

| class | rendered as | meaning |
|---|---|---|
| `JOIN` | `""` | the two tokens are halves of one word split by a bad line break |
| `SPACE` | `" "` | ordinary separator |
| `NEWLINE` | `"\n"` | line-level break: bullet items, headings |
| `PARA` | `"\n\n"` | paragraph break |

Reconstruction interleaves the original words with predicted separators
(`src/newlinefix/gaps.py`), which gives a hard **no-hallucination guarantee** that a
seq2seq or LLM approach cannot: whatever the model does, it can only move whitespace.
It also makes the model small and fast — this is a token-classification head, not a
generative decoder.

Input newlines are deliberately **ignored** (the text is re-tokenized on all
whitespace): the problem statement says existing newlines are unreliable, and the
worst case — every newline wrong — is then identical to the average case.

### Self-supervised data

Clean, well-formatted text already contains the labels. `scripts/prepare_data.py`
streams two corpora (never downloaded in full):

- **wikimedia/wikipedia 20231101.en** — encyclopedic prose with headings.
  (`Salesforce/wikitext` raw was probed first and rejected: the "raw" variant is
  word-tokenized — ` @-@ ` joiners, split contractions — which would poison training.)
- **neuralwork/arxiver** — arXiv papers converted to markdown: headings, bullet
  lists, exactly the register of the challenge's example. Cleaning strips code
  fences, display math, and HTML.

Documents are normalized to canonical form (single spaces / `\n` / `\n\n`), filtered
(≥80 words, ≥60% English-prose characters, ≥2 structural gaps), exact-deduplicated,
and split **by content hash** (stable train/val/test membership). Corruption
(`src/newlinefix/corruption.py`) then destroys newlines and randomly splits ~2% of
words mid-word (`JOIN` supervision).

### Models compared

| model | what it is | params |
|---|---|---|
| `majority` | every gap is a SPACE | — |
| `rules` | newline before bullet chars, break before `3.2.3`-style tokens | — |
| `scratch` | byte-level 2-layer BiLSTM, trained only on our data | 2.4M |
| `encoder` | **fine-tuned `distilroberta-base`** token classifier (served model) | 82M |

Both neural models read a window of words and classify every gap in it; long inputs
are handled by overlapping sliding windows stitched on their central regions
(`src/newlinefix/predict.py`). Training uses class-weighted cross-entropy
(SPACE is ~94% of gaps; weights `clip(sqrt(N/4c), 0.25, 20)`), AdamW, warmup + decay,
best-checkpoint selection by validation macro-F1 over {JOIN, NEWLINE, PARA}, and
periodic mid-epoch checkpoints so interrupted runs still leave a servable model.

### Evaluation protocol

Held-out test documents are corrupted with a fixed seed; every model re-predicts all
gaps of the same inputs. Metrics (`src/newlinefix/metrics.py`):

- **per-class precision/recall/F1** and **macro-F1 over {JOIN, NEWLINE, PARA}** —
  the primary quality number (SPACE is excluded as the trivial majority class);
- **break-F1** — binary "should any break go here", forgiving NEWLINE/PARA confusion;
- **Pk / WindowDiff** — standard text-segmentation error rates over paragraph
  boundaries (lower is better; documents too short to probe are excluded, not
  scored as perfect);
- **edit similarity** — 1 − char-edit-distance / length, computed exactly via a
  per-character-boundary decomposition that stays valid under JOIN mistakes;
- **exact-match rate**, **words/sec throughput**, and **latency per document**.

Everything is reproducible: `scripts/evaluate.py --seed 13` writes
`results/eval_results.{json,md}` including the challenge's README example fixed by
every model.

## Results

<!-- RESULTS_TABLE -->

## Conclusion

<!-- CONCLUSION -->

## Engineering notes

- **Quality gates**: 101 unit/property tests (`pytest`), `ruff` lint + format, `ty`
  type checking. Round-trip and label-alignment invariants are property-tested; the
  windowed-stitching logic is tested against an index oracle across window sizes.
- **Adversarial review**: before training, the codebase went through a 53-agent
  review — five specialized reviewers, every finding re-verified by three
  independent skeptics. 15 confirmed findings (metric-validity bugs, corpus-cleaning
  edge cases, a batch-composition dependence in the BiLSTM) were fixed and are
  visible in the git history.
- **Failure modes & limitations**: word-merge errors (`inour` → `in our`) are out of
  scope for the gap framing (would need intra-word split points); NEWLINE vs PARA
  confusion is the dominant residual error class; corpus conventions (Wikipedia +
  arXiv) bias what "correct" structure means.
- **What I'd do next**: train-step resume (optimizer-state checkpoints), ONNX/int8
  export for faster CPU serving, a hard-negative corruption curriculum, and a
  word-merge repair head.
