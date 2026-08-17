"""Fine-tune a pretrained encoder for token-gap classification (manual training loop).

Reads canonical documents from --data/{train,val}.jsonl, corrupts them into
(words, gap-labels) windows via the newlinefix data pipeline, and trains an
AutoModelForTokenClassification head with class-weighted cross-entropy. The best
epoch by macro-F1 over the structural classes {JOIN, NEWLINE, PARA} is saved to
--out in a format EncoderGapPredictor.load can serve.
"""

import json
import math
import random
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import torch
import typer
from pydantic import BaseModel
from rich.console import Console
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from newlinefix.data import Window, class_weights, load_training_windows
from newlinefix.gaps import GAP_LABELS, JOIN, NEWLINE, NUM_GAP_CLASSES, PARA
from newlinefix.metrics import accuracy, confusion_matrix, macro_f1, per_class_prf
from newlinefix.models.encoder import EncoderGapPredictor, last_subtoken_positions, pick_device

STRUCTURAL_CLASSES = (JOIN, NEWLINE, PARA)

console = Console()


class TrainConfig(BaseModel):
    """All knobs of one training run, as parsed from the CLI."""

    data: Path
    model_name: str
    out: Path
    max_words: int
    epochs: int
    batch_size: int
    lr: float
    warmup_frac: float
    weight_decay: float
    train_windows: int | None
    val_windows: int
    seed: int
    device: str | None
    log_every: int
    checkpoint_every: int


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
    cm = confusion_matrix(y_true.tolist(), y_pred.tolist())
    per_class = per_class_prf(cm)
    metrics: dict[str, float] = {
        "accuracy": accuracy(cm),
        "macro_f1_structural": macro_f1(cm, STRUCTURAL_CLASSES),
    }
    for name, m in per_class.items():
        metrics[f"precision_{name}"] = m["precision"]
        metrics[f"recall_{name}"] = m["recall"]
        metrics[f"f1_{name}"] = m["f1"]
    return metrics


def write_training_log(out: Path, steps: list[dict], epochs: list[dict]) -> None:
    """Loss/metric history written alongside the checkpoint for later inspection."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_log.json").write_text(
        json.dumps({"steps": steps, "epochs": epochs}, indent=2) + "\n", encoding="utf-8"
    )


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TrainConfig,
    val_metrics: dict[str, float],
) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.out)
    tokenizer.save_pretrained(cfg.out)
    config = {
        "max_words": cfg.max_words,
        "overlap": EncoderGapPredictor.overlap,
        "model_name": cfg.model_name,
        "val_metrics": val_metrics,
    }
    (cfg.out / "predictor_config.json").write_text(json.dumps(config, indent=2) + "\n")


def main(
    data: Path = Path("data/docs"),
    model_name: str = "distilroberta-base",
    out: Path = Path("artifacts/encoder"),
    max_words: int = 180,
    epochs: int = 2,
    batch_size: int = 32,
    # Default from the lr sweep in report.md: {2e-5: 0.68, 5e-5: 0.76, 1e-4: 0.78, 2e-4: 0.77}
    # val macro-F1 at the 8k-window budget.
    lr: float = 1e-4,
    warmup_frac: float = 0.06,
    weight_decay: float = 0.01,
    train_windows: int | None = None,
    val_windows: int = 2000,
    seed: int = 42,
    device: Annotated[str | None, typer.Option(help="default: auto mps > cuda > cpu")] = None,
    log_every: int = 50,
    checkpoint_every: Annotated[
        int,
        typer.Option(
            help="also save a serving checkpoint every N steps (0 = epoch end only), "
            "so an interrupted run still leaves a usable model"
        ),
    ] = 300,
) -> None:
    """Fine-tune a pretrained encoder for token-gap classification."""
    cfg = TrainConfig(
        data=data,
        model_name=model_name,
        out=out,
        max_words=max_words,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_frac=warmup_frac,
        weight_decay=weight_decay,
        train_windows=train_windows,
        val_windows=val_windows,
        seed=seed,
        device=device,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
    )
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train_set = load_training_windows(
        cfg.data / "train.jsonl", cfg.max_words, cfg.seed, limit=cfg.train_windows
    )
    val_set = load_training_windows(
        cfg.data / "val.jsonl", cfg.max_words, cfg.seed + 1, limit=cfg.val_windows
    )
    if not train_set or not val_set:
        raise SystemExit(f"empty train or val windows loaded from {cfg.data}")
    console.print(f"train windows: {len(train_set)}  val windows: {len(val_set)}  device: {device}")

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(cfg.model_name, add_prefix_space=True),
    )
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=NUM_GAP_CLASSES,
        id2label=dict(enumerate(GAP_LABELS)),
        label2id={name: i for i, name in enumerate(GAP_LABELS)},
    )
    model.to(device)

    weights = class_weights(train_set).to(device)
    console.print(f"class weights: {[round(float(w), 3) for w in weights]}", markup=False)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, weight=weights)

    no_decay = ("bias", "LayerNorm", "layer_norm")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.lr,
    )
    steps_per_epoch = math.ceil(len(train_set) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))
    # Linear warmup to the peak lr, then linear decay to zero (same math as the
    # previous hand-written lambda).
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    rng = random.Random(cfg.seed)
    log_steps: list[dict] = []
    log_epochs: list[dict] = []
    best_macro = -1.0
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = list(range(len(train_set)))
        rng.shuffle(order)
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(range(0, len(order), cfg.batch_size), desc=f"epoch {epoch}/{cfg.epochs}")
        for batch_start in pbar:
            batch = [train_set[i] for i in order[batch_start : batch_start + cfg.batch_size]]
            enc, labels = encode_batch(tokenizer, batch)
            logits = model(**enc.to(device)).logits
            loss = loss_fn(logits.view(-1, NUM_GAP_CLASSES), labels.view(-1).to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise SystemExit(
                    f"non-finite training loss at step {step}; aborting instead of "
                    "saving a corrupt checkpoint (check lr / device numerics)"
                )
            running_loss += loss_value
            running_n += 1
            if step % cfg.log_every == 0:
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
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                save_checkpoint(model, tokenizer, cfg, {"mid_epoch_step": float(step)})
                write_training_log(cfg.out, log_steps, log_epochs)

        metrics = evaluate(model, tokenizer, val_set, cfg.batch_size, device)
        log_epochs.append({"epoch": epoch, **metrics})
        write_training_log(cfg.out, log_steps, log_epochs)
        f1_str = " ".join(f"{name}={metrics[f'f1_{name}']:.3f}" for name in GAP_LABELS)
        console.print(
            f"epoch {epoch}: val acc={metrics['accuracy']:.4f} "
            f"macro-F1(JOIN,NEWLINE,PARA)={metrics['macro_f1_structural']:.4f} | {f1_str}"
        )
        if metrics["macro_f1_structural"] > best_macro:
            best_macro = metrics["macro_f1_structural"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, tokenizer, cfg, metrics)
            console.print(f"  [green]new best; checkpoint saved to {cfg.out}[/green]")

    # Later mid-epoch checkpoints may have overwritten the best epoch's save;
    # restore the best weights and make them the final on-disk checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(model, tokenizer, cfg, best_metrics)
    f1_str = " ".join(f"{name}={best_metrics[f'f1_{name}']:.3f}" for name in GAP_LABELS)
    console.print(
        f"[bold]BEST epoch={best_epoch} "
        f"macro-F1(JOIN,NEWLINE,PARA)={best_macro:.4f} | {f1_str}[/bold]"
    )


if __name__ == "__main__":
    typer.run(main)
