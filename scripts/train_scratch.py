"""Train the from-scratch byte-level BiLSTM gap classifier.

Manual training loop: weighted cross-entropy over logits gathered at gap byte
positions, AdamW with linear warmup + cosine decay, gradient clipping, and the
best checkpoint kept by macro-F1 over the structural classes {JOIN, NEWLINE,
PARA}. Saves model.pt (state_dict) + config.json into --out.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from newlinefix.data import Window, load_training_windows
from newlinefix.distill import EncoderTeacher, distillation_loss
from newlinefix.gaps import GAP_LABELS, JOIN, NEWLINE, NUM_GAP_CLASSES, PARA
from newlinefix.models.scratch import (
    MAX_BYTES,
    CharBiLSTM,
    ScratchGapPredictor,
    encode_window,
    gap_byte_positions,
    resolve_device,
)

Batch = tuple[Tensor, Tensor, Tensor]

IGNORE_INDEX = -100
STRUCTURAL_CLASSES = (JOIN, NEWLINE, PARA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/docs"), help="dir with train.jsonl and val.jsonl"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/scratch"))
    parser.add_argument("--max-words", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-windows", type=int, default=None)
    parser.add_argument("--val-windows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="also save a serving checkpoint every N steps (0 = end only), "
        "so an interrupted run still leaves a usable model",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="dir with a fine-tuned encoder (scripts/train_encoder.py output); "
        "when set, adds a knowledge-distillation loss over its per-gap soft targets",
    )
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="KD loss weight in [0,1]")
    parser.add_argument("--distill-temp", type=float, default=2.0, help="KD softmax temperature")
    return parser.parse_args()


def collate(batch: list[Window]) -> Batch:
    """(ids [B, L], gap positions [B, G], labels [B, G]); labels padded with -100.

    Byte ids are zero-padded to the batch max (capped at MAX_BYTES); gaps whose
    space position falls beyond the cap are masked out of the loss.
    """
    per_window: list[tuple[list[int], list[int], list[int]]] = []
    for words, labels in batch:
        ids = encode_window(words)[:MAX_BYTES]
        positions: list[int] = []
        kept: list[int] = []
        for pos, label in zip(gap_byte_positions(words), labels, strict=True):
            if pos < len(ids):
                positions.append(pos)
                kept.append(label)
        per_window.append((ids, positions, kept))
    max_len = max(len(ids) for ids, _, _ in per_window)
    max_gaps = max(1, max(len(pos) for _, pos, _ in per_window))
    ids_t = torch.zeros(len(per_window), max_len, dtype=torch.long)
    pos_t = torch.zeros(len(per_window), max_gaps, dtype=torch.long)
    lab_t = torch.full((len(per_window), max_gaps), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, positions, kept) in enumerate(per_window):
        ids_t[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        if positions:
            pos_t[row, : len(positions)] = torch.tensor(positions, dtype=torch.long)
            lab_t[row, : len(kept)] = torch.tensor(kept, dtype=torch.long)
    return ids_t, pos_t, lab_t


def save_artifacts(
    out: Path,
    model: CharBiLSTM,
    state: dict[str, Tensor],
    max_words: int,
    overlap: int,
    extra: dict[str, Any],
) -> None:
    """Write model.pt + config.json in the layout ScratchGapPredictor.load expects."""
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "model.pt")
    config: dict[str, Any] = {
        **model.config(),
        "max_words": max_words,
        "overlap": overlap,
        "max_bytes": MAX_BYTES,
        **extra,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def write_training_log(out: Path, steps: list[dict], epochs: list[dict]) -> None:
    """Loss/metric history written alongside the checkpoint for later inspection."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_log.json").write_text(
        json.dumps({"steps": steps, "epochs": epochs}, indent=2) + "\n", encoding="utf-8"
    )


def gather_gap_logits(logits: Tensor, positions: Tensor) -> Tensor:
    """logits [B, L, C], positions [B, G] -> gap logits [B, G, C]."""
    index = positions.unsqueeze(-1).expand(-1, -1, logits.size(-1))
    return logits.gather(1, index)


def class_weights(windows: list[Window]) -> Tensor:
    """w_c = clip((N_total / (4 * count_c)) ** 0.5, 0.25, 20.0)."""
    counts = np.zeros(NUM_GAP_CLASSES, dtype=np.float64)
    for _, labels in windows:
        counts += np.bincount(labels, minlength=NUM_GAP_CLASSES)
    total = counts.sum()
    weights = np.sqrt(total / (NUM_GAP_CLASSES * np.maximum(counts, 1.0)))
    return torch.tensor(np.clip(weights, 0.25, 20.0), dtype=torch.float32)


def prf_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Masked accuracy plus per-class precision/recall/F1 and structural macro-F1."""
    per_class: dict[str, dict[str, float | int]] = {}
    f1_by_class: list[float] = []
    for cls_id, name in enumerate(GAP_LABELS):
        tp = float(np.sum((preds == cls_id) & (labels == cls_id)))
        fp = float(np.sum((preds == cls_id) & (labels != cls_id)))
        fn = float(np.sum((preds != cls_id) & (labels == cls_id)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.sum(labels == cls_id)),
        }
        f1_by_class.append(f1)
    accuracy = float(np.mean(preds == labels)) if labels.size else 0.0
    macro_f1 = float(np.mean([f1_by_class[c] for c in STRUCTURAL_CLASSES]))
    return {"accuracy": accuracy, "macro_f1_structural": macro_f1, "per_class": per_class}


def evaluate(model: CharBiLSTM, batches: list[Batch], device: torch.device) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    preds_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for ids, positions, labels in batches:
            logits = gather_gap_logits(model(ids.to(device)), positions.to(device))
            preds = logits.argmax(dim=-1).cpu()
            mask = labels != IGNORE_INDEX
            preds_parts.append(preds[mask].numpy())
            label_parts.append(labels[mask].numpy())
    if was_training:
        model.train()
    preds = np.concatenate(preds_parts) if preds_parts else np.zeros(0, dtype=np.int64)
    labels = np.concatenate(label_parts) if label_parts else np.zeros(0, dtype=np.int64)
    return prf_metrics(preds, labels)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_windows = load_training_windows(
        args.data / "train.jsonl", args.max_words, args.seed, limit=args.train_windows
    )
    val_windows = load_training_windows(
        args.data / "val.jsonl", args.max_words, args.seed + 1, limit=args.val_windows
    )
    if not train_windows or not val_windows:
        raise SystemExit(f"empty train or val windows loaded from {args.data}")
    print(f"train windows: {len(train_windows)}  val: {len(val_windows)}  device: {device}")

    model = CharBiLSTM(args.embed_size, args.hidden_size, args.layers, args.dropout).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights = class_weights(train_windows).to(device)
    named = dict(zip(GAP_LABELS, [round(w, 3) for w in weights.tolist()], strict=True))
    print(f"class weights: {named}")
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    teacher: EncoderTeacher | None = None
    distill_extra: dict[str, Any] = {}
    if args.teacher is not None:
        teacher = EncoderTeacher(args.teacher, device)
        distill_extra = {
            "teacher": str(args.teacher),
            "distill_alpha": args.distill_alpha,
            "distill_temp": args.distill_temp,
        }
        print(
            f"distilling from {args.teacher} "
            f"(alpha={args.distill_alpha}, temperature={args.distill_temp})"
        )

    steps_per_epoch = math.ceil(len(train_windows) / args.batch_size)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_batches = [
        collate(val_windows[i : i + args.batch_size])
        for i in range(0, len(val_windows), args.batch_size)
    ]

    if args.max_words < 8:
        raise SystemExit("--max-words must be >= 8 so windowed inference stays valid")
    # Keep the stitching overlap valid (even, >= 2, < max_words) for small --max-words runs.
    overlap = max(2, min(ScratchGapPredictor.overlap, 2 * (args.max_words // 4)))
    epoch_rng = random.Random(args.seed)
    log_steps: list[dict] = []
    log_epochs: list[dict] = []
    best_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] = {}
    best_epoch = 0
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_windows)))
        epoch_rng.shuffle(order)
        running = 0.0
        since_log = 0
        for start in range(0, len(order), args.batch_size):
            batch = [train_windows[j] for j in order[start : start + args.batch_size]]
            ids, positions, labels = collate(batch)
            logits = gather_gap_logits(model(ids.to(device)), positions.to(device))
            loss = loss_fn(logits.reshape(-1, NUM_GAP_CLASSES), labels.reshape(-1).to(device))
            if teacher is not None:
                t_logits, t_valid = teacher.gap_logits(batch, logits.size(1))
                kd = distillation_loss(
                    logits, t_logits, t_valid, labels.to(device), args.distill_temp
                )
                loss = (1.0 - args.distill_alpha) * loss + args.distill_alpha * kd
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise SystemExit(
                    f"non-finite training loss at step {step}; aborting instead of "
                    "saving a corrupt checkpoint (check lr / device numerics)"
                )
            running += loss_value
            since_log += 1
            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {running / since_log:.4f} lr {lr_now:.2e}"
                )
                log_steps.append({"step": step, "loss": running / since_log, "lr": lr_now})
                running = 0.0
                since_log = 0
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_artifacts(
                    args.out,
                    model,
                    snapshot,
                    args.max_words,
                    overlap,
                    {"mid_epoch_step": step, "epoch": epoch, **distill_extra},
                )
                write_training_log(args.out, log_steps, log_epochs)
        metrics = evaluate(model, val_batches, device)
        macro_f1 = float(metrics["macro_f1_structural"])
        print(
            f"epoch {epoch} val acc {metrics['accuracy']:.4f} "
            f"macro-F1(JOIN/NEWLINE/PARA) {macro_f1:.4f}"
        )
        for name, m in metrics["per_class"].items():
            print(
                f"  {name:<8} P {m['precision']:.3f} R {m['recall']:.3f} "
                f"F1 {m['f1']:.3f} n={m['support']}"
            )
        log_epochs.append(
            {
                "epoch": epoch,
                "accuracy": metrics["accuracy"],
                "macro_f1_structural": macro_f1,
                **{f"f1_{name}": m["f1"] for name, m in metrics["per_class"].items()},
            }
        )
        write_training_log(args.out, log_steps, log_epochs)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    save_artifacts(
        args.out,
        model,
        best_state,
        args.max_words,
        overlap,
        {"best_epoch": best_epoch, "val_metrics": best_metrics, **distill_extra},
    )
    print(f"saved best checkpoint (epoch {best_epoch}, macro-F1 {best_f1:.4f}) to {args.out}")


if __name__ == "__main__":
    main()
