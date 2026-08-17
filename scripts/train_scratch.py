"""Train the from-scratch byte-level BiLSTM gap classifier.

Manual training loop: weighted cross-entropy over logits gathered at gap byte
positions, AdamW with linear warmup + cosine decay, gradient clipping, and the
best checkpoint kept by macro-F1 over the structural classes {JOIN, NEWLINE,
PARA}. Saves model.pt (state_dict) + config.json into --out.
"""

import json
import math
import random
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from pydantic import BaseModel
from rich.console import Console
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

console = Console()


class TrainConfig(BaseModel):
    """All knobs of one training run, as parsed from the CLI."""

    data: Path
    out: Path
    max_words: int
    epochs: int
    batch_size: int
    lr: float
    warmup_frac: float
    embed_size: int
    hidden_size: int
    layers: int
    dropout: float
    train_windows: int | None
    val_windows: int
    seed: int
    device: str
    log_every: int
    checkpoint_every: int
    teacher: Path | None
    distill_alpha: float
    distill_temp: float


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


@click.command(help=__doc__)
@click.option(
    "--data",
    type=click.Path(path_type=Path),
    default=Path("data/docs"),
    show_default=True,
    help="dir with train.jsonl and val.jsonl",
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("artifacts/scratch"), show_default=True
)
@click.option("--max-words", type=int, default=150, show_default=True)
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--warmup-frac", type=float, default=0.03, show_default=True)
@click.option("--embed-size", type=int, default=64, show_default=True)
@click.option("--hidden-size", type=int, default=256, show_default=True)
@click.option("--layers", type=int, default=2, show_default=True)
@click.option("--dropout", type=float, default=0.2, show_default=True)
@click.option("--train-windows", type=int, default=None)
@click.option("--val-windows", type=int, default=2000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--log-every", type=int, default=100, show_default=True)
@click.option(
    "--checkpoint-every",
    type=int,
    default=500,
    show_default=True,
    help="also save a serving checkpoint every N steps (0 = end only), "
    "so an interrupted run still leaves a usable model",
)
@click.option(
    "--teacher",
    type=click.Path(path_type=Path),
    default=None,
    help="dir with a fine-tuned encoder (scripts/train_encoder.py output); "
    "when set, adds a knowledge-distillation loss over its per-gap soft targets",
)
@click.option(
    "--distill-alpha", type=float, default=0.5, show_default=True, help="KD loss weight in [0,1]"
)
@click.option(
    "--distill-temp", type=float, default=2.0, show_default=True, help="KD softmax temperature"
)
def main(**kwargs: Any) -> None:
    cfg = TrainConfig(**kwargs)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_windows = load_training_windows(
        cfg.data / "train.jsonl", cfg.max_words, cfg.seed, limit=cfg.train_windows
    )
    val_windows = load_training_windows(
        cfg.data / "val.jsonl", cfg.max_words, cfg.seed + 1, limit=cfg.val_windows
    )
    if not train_windows or not val_windows:
        raise SystemExit(f"empty train or val windows loaded from {cfg.data}")
    console.print(f"train windows: {len(train_windows)}  val: {len(val_windows)}  device: {device}")

    model = CharBiLSTM(cfg.embed_size, cfg.hidden_size, cfg.layers, cfg.dropout).to(device)
    console.print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights = class_weights(train_windows).to(device)
    named = dict(zip(GAP_LABELS, [round(w, 3) for w in weights.tolist()], strict=True))
    console.print(f"class weights: {named}", markup=False)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    teacher: EncoderTeacher | None = None
    distill_extra: dict[str, Any] = {}
    if cfg.teacher is not None:
        teacher = EncoderTeacher(cfg.teacher, device)
        distill_extra = {
            "teacher": str(cfg.teacher),
            "distill_alpha": cfg.distill_alpha,
            "distill_temp": cfg.distill_temp,
        }
        console.print(
            f"distilling from {cfg.teacher} "
            f"(alpha={cfg.distill_alpha}, temperature={cfg.distill_temp})"
        )

    steps_per_epoch = math.ceil(len(train_windows) / cfg.batch_size)
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_batches = [
        collate(val_windows[i : i + cfg.batch_size])
        for i in range(0, len(val_windows), cfg.batch_size)
    ]

    if cfg.max_words < 8:
        raise SystemExit("--max-words must be >= 8 so windowed inference stays valid")
    # Keep the stitching overlap valid (even, >= 2, < max_words) for small --max-words runs.
    overlap = max(2, min(ScratchGapPredictor.overlap, 2 * (cfg.max_words // 4)))
    epoch_rng = random.Random(cfg.seed)
    log_steps: list[dict] = []
    log_epochs: list[dict] = []
    best_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] = {}
    best_epoch = 0
    step = 0
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(train_windows)))
        epoch_rng.shuffle(order)
        running = 0.0
        since_log = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_windows[j] for j in order[start : start + cfg.batch_size]]
            ids, positions, labels = collate(batch)
            logits = gather_gap_logits(model(ids.to(device)), positions.to(device))
            loss = loss_fn(logits.reshape(-1, NUM_GAP_CLASSES), labels.reshape(-1).to(device))
            if teacher is not None:
                t_logits, t_valid = teacher.gap_logits(batch, logits.size(1))
                kd = distillation_loss(
                    logits, t_logits, t_valid, labels.to(device), cfg.distill_temp
                )
                loss = (1.0 - cfg.distill_alpha) * loss + cfg.distill_alpha * kd
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
            if step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                console.print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {running / since_log:.4f} lr {lr_now:.2e}"
                )
                log_steps.append({"step": step, "loss": running / since_log, "lr": lr_now})
                running = 0.0
                since_log = 0
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_artifacts(
                    cfg.out,
                    model,
                    snapshot,
                    cfg.max_words,
                    overlap,
                    {"mid_epoch_step": step, "epoch": epoch, **distill_extra},
                )
                write_training_log(cfg.out, log_steps, log_epochs)
        metrics = evaluate(model, val_batches, device)
        macro_f1 = float(metrics["macro_f1_structural"])
        console.print(
            f"epoch {epoch} val acc {metrics['accuracy']:.4f} "
            f"macro-F1(JOIN/NEWLINE/PARA) {macro_f1:.4f}"
        )
        for name, m in metrics["per_class"].items():
            console.print(
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
        write_training_log(cfg.out, log_steps, log_epochs)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    save_artifacts(
        cfg.out,
        model,
        best_state,
        cfg.max_words,
        overlap,
        {"best_epoch": best_epoch, "val_metrics": best_metrics, **distill_extra},
    )
    console.print(
        f"[green]saved best checkpoint (epoch {best_epoch}, "
        f"macro-F1 {best_f1:.4f}) to {cfg.out}[/green]"
    )


if __name__ == "__main__":
    main()
