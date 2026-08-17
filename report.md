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
- The served model is a **fine-tuned `distilroberta-base`** token classifier
  (published: [preneond/newlinefix-encoder](https://huggingface.co/preneond/newlinefix-encoder)).
  It is compared against a **from-scratch byte-level BiLSTM** (~2.4M params) and two
  non-neural baselines — see [Results](#results).

## How to run

```bash
# environment (Python 3.14, uv)
uv sync

# tests, lint, types (poe check = lint + typecheck + test, same as CI)
uv run poe check

# 1. data: stream + clean corpora into canonical documents (train/val/test JSONL)
uv run python scripts/prepare_data.py --out data/docs --wikitext-docs 12000 --markdown-docs 12000

# 2. train the two models (auto-selects MPS/CUDA/CPU)
uv run python scripts/train_encoder.py --data data/docs --out artifacts/encoder --epochs 2 --train-windows 80000
uv run python scripts/train_scratch.py --data data/docs --out artifacts/scratch --epochs 3

# 3. evaluate all models on the held-out test split
uv run python scripts/evaluate.py --models all --out results

# publish the trained model to the Hugging Face Hub (repo id then works as a
# model source everywhere a local artifact dir does, incl. NEWLINEFIX_MODEL_DIR)
uv run poe publish --repo-id preneond/newlinefix-encoder

# HTTP API (POST /fix {"text": ...} -> fixed text); --model picks any artifact
# dir or HF Hub repo id, --list-models shows the servable local checkpoints
uv run poe serve
# ... or containerized (CPU-only torch; bakes the trained model into the image)
docker build -t newlinefix . && docker run -p 8000:8000 newlinefix
# ... or the whole stack: API on :8000 plus the UI on :8501, talking to it over HTTP
docker compose up

# minimal UI
uv run poe streamlit
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

Held-out test split (120 documents, ~205k words, seed 13; corrupted inputs, canonical
targets). Macro-F1 is over {JOIN, NEWLINE, PARA}; Pk/WindowDiff lower is better.

| Model | Gap acc | Macro-F1 | Break-F1 | Pk | WinDiff | EditSim | Words/s | ms/doc |
|---|---|---|---|---|---|---|---|---|
| majority | 0.957 | 0.000 | 0.000 | 0.341 | 0.341 | 0.986 | ~10⁹ | 0.00 |
| rules | 0.954 | 0.034 | 0.040 | 0.348 | 0.352 | 0.986 | ~10⁷ | 0.19 |
| **encoder** | **0.981** | **0.815** | **0.769** | **0.240** | **0.311** | **0.994** | 10,288 | 185 |
| scratch | 0.959 | 0.632 | 0.617 | 0.365 | 0.485 | 0.987 | 2,801 | 678 |

The served encoder is the lr=1e-4 fine-tune selected by the learning-rate sweep in
[Model exploration](#model-exploration). Training budgets were deliberately small
(single epoch, ~6–16 minutes each on an Apple M4 Max via MPS): encoder 8k windows,
scratch 30k windows. Validation macro-F1 scaled 0.53 → 0.78 when the encoder's data
grew 2k → 8k windows, so there is clear headroom; the full corpus is ~260k windows.
(Words/s figures vary ±30% run-to-run with machine load; quality metrics are exactly
reproducible via `--seed`.)

On the challenge's own example, the encoder reproduces the expected output almost
exactly — heading isolated, paragraph break placed, `• bullet` on its own line, and
`que\nries` repaired to `queries` (its one deviation: a blank line instead of a single
newline after "ways:", the NEWLINE↔PARA confusion that dominates its residual errors).
Full per-model outputs: `results/eval_results.md`.

## Model exploration

Beyond the headline comparison, three further axes were explored under the same
protocol (merged table: `results/exploration_results.md`). Val is validation
macro-F1 over {JOIN, NEWLINE, PARA}; test metrics are on the held-out 120-doc split.

### Learning-rate sweep (distilroberta-base, 8k windows, 1 epoch)

| lr | val macro-F1 | test macro-F1 | test break-F1 | test Pk |
|---|---|---|---|---|
| 2e-5 | 0.682 | 0.729 | 0.696 | 0.299 |
| 5e-5 | 0.760 | 0.789 | 0.748 | 0.249 |
| **1e-4** | **0.784** | **0.815** | **0.769** | **0.240** |
| 2e-4 | 0.774 | 0.809 | 0.760 | 0.250 |

The conventional 5e-5 fine-tuning rate left ~2.5 points on the table: with a single
epoch over 8k windows the head benefits from a hotter schedule, and quality only
starts regressing past 1e-4. **The lr=1e-4 model was promoted to be the served
encoder** (and the trainer's new default) after verifying it fixes the README
example identically to the previous model.

### Distillation: encoder teacher → byte-BiLSTM student

The fine-tuned encoder was frozen as a teacher and the 2.4M-param BiLSTM retrained
with per-gap soft targets: `loss = (1−α)·CE(hard) + α·τ²·KL(student/τ ‖ teacher/τ)`
with τ = 2 (`src/newlinefix/distill.py`; `scripts/train_scratch.py --teacher`).
Same data budget as the scratch baseline (30k windows, 1 epoch).

| student (2.4M params, identical architecture) | test macro-F1 | test break-F1 | test Pk |
|---|---|---|---|
| scratch (α = 0 — hard labels only) | 0.632 | 0.617 | 0.365 |
| **distilled, α = 0.5** | **0.675** | **0.657** | **0.323** |
| distilled, α = 0.9* | 0.634 | 0.627 | 0.322 |

\* the α = 0.9 run was interrupted at step 400/469 and scored from its last
checkpoint; the LR had already decayed to near zero, so the ranking is reliable.

Balanced hard+soft targets buy **+4.3 test macro-F1 at identical size and speed**,
recovering about a quarter of the gap to the 34×-larger teacher. The teacher's soft
targets carry inter-class structure a hard label can't ("this gap is NEWLINE-or-PARA,
certainly not SPACE"). Going nearly all-soft (α = 0.9) gives the gain back — the
hard-label term still anchors the decision boundary for the rare classes.

### Backbone sweep (8k windows, 1 epoch, lr=1e-4)

`train_encoder.py --model-name` fine-tunes any HF token-classification backbone
under the served recipe; four backbones spanning 14M–125M parameters trace the
size/quality/latency frontier (full metrics: `results/explore-backbones/`). Words/s
come from one eval run on an M4 MacBook Air — a different machine than the headline
table's M4 Max — so only the relative ordering is meaningful.

| backbone | params | val macro-F1 | test macro-F1 | test break-F1 | test Pk | words/s |
|---|---|---|---|---|---|---|
| electra-small-discriminator | 14M | 0.649 | 0.702 | 0.665 | 0.282 | 4,727 |
| distilbert-base-cased | 66M | 0.723 | 0.763 | 0.710 | 0.278 | 1,974 |
| **distilroberta-base (served)** | **82M** | **0.783** | **0.814** | **0.769** | **0.229** | 2,044 |
| roberta-base\* | 125M | 0.817 | 0.825 | 0.783 | 0.209 | 1,039 |

Quality scales monotonically with size, but with sharply diminishing returns around
the served model: roberta-base buys +1.1 test macro-F1 over distilroberta for 1.5×
the parameters and ~2× the latency, while electra-small gives up 11 points for its
6× smaller footprint. electra-small is also *uncased*, and it pays exactly where
capitalization carries the signal: NEWLINE (headings, bullets) is its weakest class
relative to the cased distilbert (F1 0.685 vs 0.772). **No backbone replaces the
served model** — distilroberta-base keeps the best quality-per-millisecond, and
roberta-base's margin is well short of the "clearly wins on quality" bar that
promoted the lr=1e-4 fine-tune.

\* roberta-base could not be fine-tuned on Apple MPS (reproduced on torch 2.13.0
*and* 2.12.1, so no released version avoids it): training
collapsed to NaN within 50 steps at every learning rate tried — loss and gradients
finite but weights NaN after the AdamW step, with `clip_grad_norm_` returning
impossibly small total norms; per-step syncs masked the failure, implicating an
async command-buffer race in the MPS backend. The row above was trained on CPU with
the identical recipe (~2 h). The smaller backbones train on MPS without issue, and
both trainers now abort on a non-finite loss instead of saving a corrupt checkpoint.

### What each technique is for

- **Fine-tuned encoder, lr-swept** — the quality/throughput sweet spot; the served model.
- **Distilled BiLSTM** — for targets that can't take an 82M-param transformer
  (CPU-only edge, strict memory): distillation upgrades the tiny model for free at
  inference time. The same recipe would distill into a smaller transformer as well.
- **Backbone sweep** — maps the size/quality/latency frontier around the served
  model: quality tracks parameter count, but nothing beats distilroberta-base on
  quality-per-millisecond, so it stays the served backbone.

## Conclusion

**Did we use an existing model? Yes — the winner is a fine-tune of one.** The served
model is `distilroberta-base` (82M params, pretrained) fine-tuned with a 4-class
token-classification head. It was compared against a from-scratch byte-level BiLSTM
(2.4M params) and two non-neural baselines, all under the same protocol.

**Fine-tuning a pretrained encoder worked best on every axis measured:**

1. **Quality** — test macro-F1 **0.815 vs 0.632** for the from-scratch model
   (break-F1 0.769 vs 0.617, Pk 0.240 vs 0.365), despite training on *4× less data*
   (8k vs 30k windows). That gap is what pretrained language knowledge buys: the
   encoder already knows "queries" is a word and where sentences break; the BiLSTM
   has to learn English from 5MB of bytes.
2. **Speed, surprisingly** — ~10k vs ~2.8k words/s. The 34×-larger transformer runs
   one parallel pass per window, while the recurrent model steps byte-by-byte
   (~1,200 sequential steps per window), which parallel hardware cannot amortize at
   batch size 1.
3. **The baselines justify the ML**: rules manage macro-F1 0.034, and the majority
   baseline's 95.7% gap *accuracy* shows why accuracy is the wrong metric here —
   the informative signals are per-class F1 and segmentation error.

Two cheap wins came out of [Model exploration](#model-exploration): a learning-rate
sweep moved the encoder 0.789 → 0.815 at zero extra cost, and knowledge distillation
from the encoder teacher moved the tiny BiLSTM 0.632 → 0.675 at identical size and
speed. A backbone sweep (14M–125M params) then confirmed the serving choice: quality
tracks size, but roberta-base tops distilroberta by only +1.1 macro-F1 at twice the
latency, so the 82M model keeps the best quality-per-millisecond and stays in
service. The from-scratch model remains a respectable result for 2.4M parameters and
minutes of training — but on this task, at this data scale, **transfer learning
dominates training from scratch on quality, data-efficiency, and (at batch-1 GPU
inference) even speed.** The lr-swept encoder is therefore the model behind the
service.

## Engineering notes

- **Quality gates**: 125 unit/property/service tests (`pytest`), `ruff` lint + format, `ty`
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
