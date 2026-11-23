"""
Full-data classification baseline training for 20 Newsgroups.

Usage:
    python src/train.py --seed 27 --data_dir data/ --results_base results/

Creates:  results/cls-baseline-seed{seed}-{jobid}/
    run_config.json, metrics.jsonl, best.pt, last.pt
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import NewsgroupDataset, load_split, load_vocab, load_config
from model import TransformerEncoder, count_params


# ── defaults matching the brief ──────────────────────────────────────────────
DEFAULTS = dict(
    embed_dim=256, num_heads=4, num_layers=4, ffn_dim=1024,
    max_len=128, dropout=0.1,
    lr=3e-4, weight_decay=0.01, batch_size=64, grad_clip=1.0,
    epochs=5, seed=27,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for tokens, pad_mask, labels in loader:
            tokens, pad_mask, labels = (
                tokens.to(device), pad_mask.to(device), labels.to(device)
            )
            logits = model(tokens, pad_mask)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(args):
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = os.path.join(
        args.results_base, f"cls-baseline-seed{args.seed}-{job_id}"
    )
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    # ── data ─────────────────────────────────────────────────────────────────
    vocab = load_vocab(args.data_dir)
    dcfg = load_config(args.data_dir)
    num_classes = dcfg["num_classes"]

    train_texts, train_labels, _ = load_split(args.data_dir, "train")
    val_texts, val_labels, _ = load_split(args.data_dir, "val")

    train_ds = NewsgroupDataset(train_texts, train_labels, vocab)
    val_ds = NewsgroupDataset(val_texts, val_labels, vocab)

    # Separate seed for data loader so batch order is independent of model seed
    g = torch.Generator(); g.manual_seed(args.seed + 1000)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    # ── model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model = TransformerEncoder(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, ffn_dim=args.ffn_dim,
        max_len=args.max_len, dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)

    n_params = count_params(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # ── run config ────────────────────────────────────────────────────────────
    run_cfg = {**vars(args), "n_params": n_params, "device": str(device),
               "job_id": job_id, "n_train": len(train_ds), "n_val": len(val_ds)}
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for tokens, pad_mask, labels in train_loader:
            tokens, pad_mask, labels = (
                tokens.to(device), pad_mask.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            loss = criterion(model(tokens, pad_mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f} | {elapsed:.1f}s")

        row = {"epoch": epoch, "train_loss": avg_loss, "val_acc": val_acc,
               "elapsed_s": elapsed}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            print(f"  → new best val: {best_val_acc:.4f}")

    best_sha = sha256_file(os.path.join(run_dir, "best.pt"))
    last_sha = sha256_file(os.path.join(run_dir, "last.pt"))
    summary = {
        "best_val_acc": best_val_acc, "n_epochs": args.epochs,
        "best_pt_sha256": best_sha, "last_pt_sha256": last_sha,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print(f"Run dir: {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--results_base", default="results/")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--grad_clip", type=float, default=DEFAULTS["grad_clip"])
    ap.add_argument("--embed_dim", type=int, default=DEFAULTS["embed_dim"])
    ap.add_argument("--num_heads", type=int, default=DEFAULTS["num_heads"])
    ap.add_argument("--num_layers", type=int, default=DEFAULTS["num_layers"])
    ap.add_argument("--ffn_dim", type=int, default=DEFAULTS["ffn_dim"])
    ap.add_argument("--max_len", type=int, default=DEFAULTS["max_len"])
    ap.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
