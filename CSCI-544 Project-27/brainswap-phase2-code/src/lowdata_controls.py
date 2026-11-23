"""
Post-hoc controls for Track A (no extra training required).

Loads the baseline checkpoint, extracts embeddings, and runs:
  knn           – cosine kNN (k chosen on val from {1,3,5,10,20})
  hopfield_ro   – softmax(β * cosine) readout (β chosen on val)

Usage:
    python src/lowdata_controls.py \\
        --baseline_dir results/lowdata-baseline-n25-seed27-<jobid>/ \\
        --seed 27 --per_class 25 \\
        --data_dir data/ --results_base results/

Outputs results/lowdata-controls-n25-seed27-<jobid>/summary.json
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import NewsgroupDataset, load_split, load_vocab, load_config
from model import TransformerEncoder, count_params
from lowdata_subsets import make_nested_subsets, make_val_subset

VAL_PER_CLASS = 200


def _extract_embeddings(model, ds, device, batch_size=256):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embs, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for tokens, pad_mask, labels in loader:
            tokens, pad_mask = tokens.to(device), pad_mask.to(device)
            z = model.encode(tokens, pad_mask)
            embs.append(z.cpu())
            labels_list.append(labels)
    return torch.cat(embs, 0), torch.cat(labels_list, 0)


def _cosine_sim(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """(Q, d) and (N, d) → (Q, N) cosine similarity matrix."""
    q_n = F.normalize(q, dim=-1)
    k_n = F.normalize(k, dim=-1)
    return q_n @ k_n.T


def knn_predict(train_emb, train_lbl, query_emb, num_classes, k):
    """Similarity-weighted vote; returns predicted labels (Q,)."""
    sim = _cosine_sim(query_emb, train_emb)   # (Q, N)
    topk_sim, topk_idx = sim.topk(k, dim=-1)  # (Q, k)
    # Weighted vote
    topk_lbl = train_lbl[topk_idx]            # (Q, k)
    votes = torch.zeros(query_emb.shape[0], num_classes)
    for q in range(query_emb.shape[0]):
        for i in range(k):
            votes[q, topk_lbl[q, i]] += topk_sim[q, i].item()
    return votes.argmax(-1)


def hopfield_readout_predict(train_emb, train_lbl, query_emb, num_classes, beta):
    """softmax(β * cosine) over one-hot memory labels."""
    sim = _cosine_sim(query_emb, train_emb)   # (Q, N)
    w = torch.softmax(beta * sim, dim=-1)     # (Q, N)
    one_hot = F.one_hot(train_lbl, num_classes).float()  # (N, C)
    logits = w @ one_hot                      # (Q, C)
    return logits.argmax(-1)


def accuracy(pred, true):
    return (pred == true).float().mean().item()


def run_controls(
    baseline_dir: str,
    seed: int,
    per_class: int,
    data_dir: str,
    results_base: str,
):
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = os.path.join(
        results_base,
        f"lowdata-controls-n{per_class}-seed{seed}-{job_id}"
    )
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab(data_dir)
    dcfg = load_config(data_dir)
    num_classes = dcfg["num_classes"]

    train_texts, train_labels, train_ids = load_split(data_dir, "train")
    val_texts, val_labels, val_ids = load_split(data_dir, "val")

    # Labeled subset (same as the training run)
    sizes_needed = [s for s in [10, 25, 100, 200] if s <= per_class]
    if per_class not in sizes_needed:
        sizes_needed.append(per_class)
    subsets = make_nested_subsets(
        train_texts, train_labels, train_ids,
        per_class_sizes=sizes_needed, seed=seed,
    )
    sub_texts, sub_labels, _ = subsets[per_class]

    val_sub_texts, val_sub_labels, _ = make_val_subset(
        val_texts, val_labels, val_ids, per_class=VAL_PER_CLASS
    )

    # Load baseline encoder
    torch.manual_seed(seed)
    encoder = TransformerEncoder(
        vocab_size=len(vocab), embed_dim=256, num_heads=4,
        num_layers=4, ffn_dim=1024, max_len=128, dropout=0.1,
        num_classes=num_classes,
    ).to(device)
    ckpt = os.path.join(baseline_dir, "best.pt")
    state = torch.load(ckpt, map_location=device)
    # Baseline checkpoint is saved from BaselineClassifier; strip "encoder." prefix
    enc_state = {k.removeprefix("encoder."): v for k, v in state.items()
                 if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    encoder.eval()

    subset_ds = NewsgroupDataset(sub_texts, sub_labels, vocab)
    val_ds = NewsgroupDataset(val_sub_texts, val_sub_labels, vocab)

    print(f"Extracting embeddings …")
    train_emb, train_lbl = _extract_embeddings(encoder, subset_ds, device)
    val_emb, val_lbl = _extract_embeddings(encoder, val_ds, device)

    results = {}

    # ── kNN ──────────────────────────────────────────────────────────────────
    K_OPTIONS = [1, 3, 5, 10, 20]
    best_k, best_k_acc = None, -1.0
    for k in K_OPTIONS:
        k_eff = min(k, len(train_emb))
        pred = knn_predict(train_emb, train_lbl, val_emb, num_classes, k_eff)
        acc = accuracy(pred, val_lbl)
        if acc > best_k_acc:
            best_k_acc, best_k = acc, k_eff
    results["knn"] = {"best_k": best_k, "val_acc": best_k_acc}
    print(f"kNN best k={best_k}, val_acc={best_k_acc:.4f}")

    # ── Hopfield readout ──────────────────────────────────────────────────────
    BETA_OPTIONS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    best_beta, best_beta_acc = None, -1.0
    for beta in BETA_OPTIONS:
        pred = hopfield_readout_predict(train_emb, train_lbl, val_emb, num_classes, beta)
        acc = accuracy(pred, val_lbl)
        if acc > best_beta_acc:
            best_beta_acc, best_beta = acc, beta
    results["hopfield_readout"] = {"best_beta": best_beta, "val_acc": best_beta_acc}
    print(f"Hopfield readout best beta={best_beta}, val_acc={best_beta_acc:.4f}")

    # Save
    with open(os.path.join(run_dir, "controls_val.json"), "w") as f:
        json.dump({"seed": seed, "per_class": per_class,
                   "baseline_dir": baseline_dir, **results}, f, indent=2)

    print(f"Controls done. Results in {run_dir}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--per_class", type=int, default=25)
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--results_base", default="results/")
    args = ap.parse_args()
    run_controls(
        args.baseline_dir, args.seed, args.per_class,
        args.data_dir, args.results_base,
    )


if __name__ == "__main__":
    main()
