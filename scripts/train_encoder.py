"""Fine-tune a pretrained encoder for token-gap classification (manual training loop).

Reads canonical documents from --data/{train,val}.jsonl, corrupts them into
(words, gap-labels) windows via the newlinefix data pipeline, and trains an
AutoModelForTokenClassification head with class-weighted cross-entropy. The best
epoch by macro-F1 over the structural classes {JOIN, NEWLINE, PARA} is saved to
--out in a format EncoderGapPredictor.load can serve.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from newlinefix.data import Window, load_training_windows
from newlinefix.gaps import GAP_LABELS, JOIN, NEWLINE, NUM_GAP_CLASSES, PARA
from newlinefix.models.encoder import EncoderGapPredictor, last_subtoken_positions, pick_device

STRUCTURAL_CLASSES = (JOIN, NEWLINE, PARA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/docs"))
    p.add_argument("--model-name", default="distilroberta-base")
    p.add_argument("--out", type=Path, default=Path("artifacts/encoder"))
    p.add_argument("--max-words", type=int, default=180)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-frac", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--train-windows", type=int, default=None)
    p.add_argument("--val-windows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="default: auto mps > cuda > cpu")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def gap_label_tensor(
    word_id_rows: list[list[int | None]], windows: list[Window], seq_len: int
) -> torch.Tensor:
    """Per-token labels: -100 everywhere except gap i at the last subtoken of word i."""
    labels = torch.full((len(windows), seq_len), -100, dtype=torch.long)
    for b, (words, gaps) in enumerate(windows):
        positions = last_subtoken_positions(word_id_rows[b], len(words))
        for i, gap in enumerate(gaps):
            pos = positions[i]
            if pos is not None:
                labels[b, pos] = gap
    return labels


def encode_batch(
    tokenizer: PreTrainedTokenizerBase, batch: list[Window]
) -> tuple[BatchEncoding, torch.Tensor]:
    enc = tokenizer(
        [words for words, _ in batch],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    word_id_rows = [enc.word_ids(batch_index=b) for b in range(len(batch))]
    labels = gap_label_tensor(word_id_rows, batch, int(enc["input_ids"].shape[1]))
    return enc, labels


def class_weights(windows: list[Window]) -> torch.Tensor:
    """w_c = clip(sqrt(N / (4 * count_c)), 0.25, 20): upweights rare JOIN, damps SPACE."""
    counts = np.zeros(NUM_GAP_CLASSES, dtype=np.float64)
    for _, gaps in windows:
        for gap in gaps:
            counts[gap] += 1
    weights = np.sqrt(counts.sum() / (NUM_GAP_CLASSES * np.maximum(counts, 1.0)))
    return torch.tensor(np.clip(weights, 0.25, 20.0), dtype=torch.float32)


def prf_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        pred_c = float(np.sum(y_pred == c))
        true_c = float(np.sum(y_true == c))
        precision[c] = tp / pred_c if pred_c else 0.0
        recall[c] = tp / true_c if true_c else 0.0
        denom = precision[c] + recall[c]
        f1[c] = 2 * precision[c] * recall[c] / denom if denom else 0.0
    return precision, recall, f1


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    windows: list[Window],
    batch_size: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    trues: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            enc, labels = encode_batch(tokenizer, windows[start : start + batch_size])
            logits = model(**enc.to(device)).logits
            batch_preds = logits.argmax(dim=-1).cpu()
            mask = labels != -100
            trues.append(labels[mask].numpy())
            preds.append(batch_preds[mask].numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    precision, recall, f1 = prf_per_class(y_true, y_pred, NUM_GAP_CLASSES)
    metrics: dict[str, float] = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1_structural": float(np.mean([f1[c] for c in STRUCTURAL_CLASSES])),
    }
    for c, name in enumerate(GAP_LABELS):
        metrics[f"precision_{name}"] = float(precision[c])
        metrics[f"recall_{name}"] = float(recall[c])
        metrics[f"f1_{name}"] = float(f1[c])
    return metrics


def write_training_log(out: Path, steps: list[dict], epochs: list[dict]) -> None:
    """Loss/metric history consumed by the marimo walkthrough notebook."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_log.json").write_text(
        json.dumps({"steps": steps, "epochs": epochs}, indent=2) + "\n", encoding="utf-8"
    )


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    out: Path,
    args: argparse.Namespace,
    val_metrics: dict[str, float],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    config = {
        "max_words": args.max_words,
        "overlap": EncoderGapPredictor.overlap,
        "model_name": args.model_name,
        "val_metrics": val_metrics,
    }
    (out / "predictor_config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    train_windows = load_training_windows(
        args.data / "train.jsonl", args.max_words, args.seed, limit=args.train_windows
    )
    val_windows = load_training_windows(
        args.data / "val.jsonl", args.max_words, args.seed + 1, limit=args.val_windows
    )
    print(f"train windows: {len(train_windows)}  val windows: {len(val_windows)}  device: {device}")

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True),
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_GAP_CLASSES,
        id2label=dict(enumerate(GAP_LABELS)),
        label2id={name: i for i, name in enumerate(GAP_LABELS)},
    )
    model.to(device)

    weights = class_weights(train_windows).to(device)
    print(f"class weights: {[round(float(w), 3) for w in weights]}")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, weight=weights)

    no_decay = ("bias", "LayerNorm", "layer_norm")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
    )
    steps_per_epoch = math.ceil(len(train_windows) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    rng = random.Random(args.seed)
    log_steps: list[dict] = []
    log_epochs: list[dict] = []
    best_macro = -1.0
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_windows)))
        rng.shuffle(order)
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(range(0, len(order), args.batch_size), desc=f"epoch {epoch}/{args.epochs}")
        for batch_start in pbar:
            batch = [train_windows[i] for i in order[batch_start : batch_start + args.batch_size]]
            enc, labels = encode_batch(tokenizer, batch)
            logits = model(**enc.to(device)).logits
            loss = loss_fn(logits.view(-1, NUM_GAP_CLASSES), labels.view(-1).to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += float(loss.item())
            running_n += 1
            if step % args.log_every == 0:
                pbar.set_postfix(
                    loss=f"{running_loss / running_n:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                log_steps.append(
                    {
                        "step": step,
                        "loss": running_loss / running_n,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                running_loss = 0.0
                running_n = 0

        metrics = evaluate(model, tokenizer, val_windows, args.batch_size, device)
        log_epochs.append({"epoch": epoch, **metrics})
        write_training_log(args.out, log_steps, log_epochs)
        f1_str = " ".join(f"{name}={metrics[f'f1_{name}']:.3f}" for name in GAP_LABELS)
        print(
            f"epoch {epoch}: val acc={metrics['accuracy']:.4f} "
            f"macro-F1(JOIN,NEWLINE,PARA)={metrics['macro_f1_structural']:.4f} | {f1_str}"
        )
        if metrics["macro_f1_structural"] > best_macro:
            best_macro = metrics["macro_f1_structural"]
            best_epoch = epoch
            best_metrics = metrics
            save_checkpoint(model, tokenizer, args.out, args, metrics)
            print(f"  new best; checkpoint saved to {args.out}")

    f1_str = " ".join(f"{name}={best_metrics[f'f1_{name}']:.3f}" for name in GAP_LABELS)
    print(f"BEST epoch={best_epoch} macro-F1(JOIN,NEWLINE,PARA)={best_macro:.4f} | {f1_str}")


if __name__ == "__main__":
    main()
