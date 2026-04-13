"""
Training and evaluation CLI for SPaR.txt.

Usage
-----
Train (default):
    python run_tagger.py

Train with custom paths / hyperparams:
    python run_tagger.py --train data/train/ --val data/val/ \
        --model-dir trained_models/ --epochs 50 --batch-size 16 --lr 0.005

Evaluate a trained model on a held-out test set:
    python run_tagger.py --evaluate --test data/test/ \
        --model-dir trained_models/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

from spar_lib.models.span_tagger import SparTagger
from spar_lib.readers.tagging_reader import (
    IDX_TO_TAG,
    TAG_TO_IDX,
    TAGS,
    SparDataset,
    collate_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_optimizer(model: SparTagger, lr: float, weight_decay: float):
    """
    AdamW with two parameter groups:
     - bias / LayerNorm weights  → weight_decay=0.05
     - everything else           → weight_decay=weight_decay
    Mirrors the original AllenNLP ``huggingface_adamw`` config.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.05},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, eps=1e-8)


def _run_epoch(
    model: SparTagger,
    loader: DataLoader,
    optimizer=None,
    scheduler=None,
    max_grad_norm: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one epoch; returns metrics dict from the model's F1 accumulator."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            gold  = batch["tags"]
            gold  = gold.to(device) if gold is not None else None

            out = model(
                ids, mask,
                gold_tags=gold,
                words=batch["words"],
                sentences=batch["sentences"],
                doc_ids=batch["doc_ids"],
            )

            if is_train and "loss" in out:
                loss = out["loss"]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

    metrics = model.get_metrics(reset=True)
    if is_train and len(loader) > 0:
        metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)

    train_ds = SparDataset(args.train, tokenizer)
    val_ds   = SparDataset(args.val,   tokenizer)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    model = SparTagger(
        num_tags=len(TAG_TO_IDX),
        label_map=IDX_TO_TAG,
        bert_model_name=args.bert_model,
        lstm_hidden_size=384,
        ffnn_hidden_size=60,
        dropout=0.05,
        freeze_bert=True,
        attention_heads=12,
    ).to(device)

    optimizer = _build_optimizer(model, lr=args.lr, weight_decay=0.1)

    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(0.1 * total_steps)          # 10 % warmup (slanted-triangular)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_f1        = -1.0
    patience_count = 0
    best_metrics   = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, optimizer=optimizer,
            scheduler=scheduler, device=device,
        )
        val_metrics = _run_epoch(model, val_loader, device=device)

        f1 = val_metrics.get("f1-measure-overall", 0.0)
        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={train_metrics.get('loss', float('nan')):.4f}  "
            f"val_f1={f1:.4f}  "
            f"val_prec={val_metrics.get('precision-overall', 0.0):.4f}  "
            f"val_rec={val_metrics.get('recall-overall', 0.0):.4f}"
        )

        if f1 > best_f1:
            best_f1        = f1
            best_metrics   = val_metrics
            patience_count = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(checkpoint, model_dir / "model.pt")
            print(f"  ✓ New best — saved to {model_dir / 'model.pt'}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping (patience={args.patience})")
                break

    print("\nBest validation metrics:")
    for k, v in sorted(best_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Save a human-readable metrics file alongside the checkpoint
    with open(model_dir / "best_metrics.json", "w") as fh:
        json.dump(best_metrics, fh, indent=2)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir  = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)

    # Re-create model with the same hyperparams used at training time
    saved_args = checkpoint.get("args", {})
    bert_model = saved_args.get("bert_model", args.bert_model)

    model = SparTagger(
        num_tags=len(TAG_TO_IDX),
        label_map=IDX_TO_TAG,
        bert_model_name=bert_model,
        lstm_hidden_size=384,
        ffnn_hidden_size=60,
        dropout=0.0,     # no dropout at eval time
        freeze_bert=True,
        attention_heads=12,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = BertTokenizerFast.from_pretrained(bert_model)
    test_ds   = SparDataset(args.test, tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Evaluating on {len(test_ds)} samples from {args.test}")

    test_metrics = _run_epoch(model, test_loader, device=device)

    print("\nTest metrics:")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(test_metrics, fh, indent=2)
        print(f"Saved to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="SPaR.txt — train / evaluate sequence tagger")

    p.add_argument("--evaluate", action="store_true",
                   help="Evaluate a trained model instead of training")

    # Data paths
    p.add_argument("--train",      default="data/train/",
                   help="Directory with training .txt/.ann pairs")
    p.add_argument("--val",        default="data/val/",
                   help="Directory with validation .txt/.ann pairs")
    p.add_argument("--test",       default="data/test/",
                   help="Directory with test .txt/.ann pairs (--evaluate only)")

    # Model / output
    p.add_argument("--model-dir",  default="trained_models/",
                   help="Where to save / load model.pt")
    p.add_argument("--bert-model", default="bert-base-cased",
                   help="HuggingFace model identifier for the BERT encoder")
    p.add_argument("--output",     default="",
                   help="(Evaluate) Write metrics JSON to this path")

    # Training hyperparams
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=0.005)
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early-stopping patience (epochs without val F1 improvement)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
