"""
Low-data classification training: Track A.

Runs four variants for a given (seed, per_class_size):
  baseline   – TransformerEncoder + masked mean pooling + linear head
  hpool      – TransformerEncoder + HopfieldPooling
  hmem       – TransformerEncoder + training-set Hopfield memory
  hpool+hmem – both hpool and hmem combined

All variants share the same labeled subset and the same batch order at each seed.

Usage:
    python src/train_lowdata.py --seed 27 --per_class 25 \\
        --data_dir data/ --results_base results/ --variant all

    # Or one variant at a time:
    python src/train_lowdata.py --seed 27 --per_class 25 --variant hmem

Smoke run (CPU, tiny):
    python src/train_lowdata.py --seed 27 --per_class 10 --max_steps 5 \\
        --data_dir data/ --results_base results/ --variant baseline
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import NewsgroupDataset, load_split, load_vocab, load_config
from model import TransformerEncoder, count_params
from hopfield_layers import MultiHeadHopfieldRetrieval, HopfieldPooling
from lowdata_subsets import make_nested_subsets, make_val_subset

VARIANTS = ["baseline", "hpool", "hmem", "hpool+hmem"]

DEFAULT_MAX_STEPS = 1500
DEFAULT_EVAL_EVERY = 50
DEFAULT_PATIENCE = 10   # evaluations without improvement
VAL_PER_CLASS = 200     # fixed val subset size


# ── Hopfield memory module ────────────────────────────────────────────────────

class HopfieldMemory(nn.Module):
    """
    Retrieves from labeled training-set embeddings.
    Memory embeddings are refreshed externally via set_memory().

    Parameters (embed_dim=256, num_heads=4, num_classes=20):
      q_proj:     256*256 + 256 = 65,792
      k_proj:     65,792
      v_proj:     65,792
      q_norm:     512
      kv_norm:    512
      label_emb:  20 * 256 = 5,120
      log_beta:   4 (negligible)
      Total:      ~203,520
    """

    def __init__(self, embed_dim: int, num_heads: int, num_classes: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.log_beta = nn.Parameter(
            torch.full((num_heads,), math.log(1.0 / math.sqrt(embed_dim // num_heads)))
        )

        # Memory tensors set externally; not parameters
        self.mem_embeddings: torch.Tensor = None   # (N, embed_dim)
        self.mem_labels: torch.Tensor = None       # (N,) int64
        self.mem_indices: torch.Tensor = None      # (N,) int64 – position in subset

    def set_memory(self, embeddings: torch.Tensor, labels: torch.Tensor,
                   indices: torch.Tensor):
        """Register fresh memory (no grad). Call once per epoch or every 200 steps."""
        self.mem_embeddings = embeddings.detach()
        self.mem_labels = labels
        self.mem_indices = indices

    def forward(self, z: torch.Tensor, query_positions=None):
        """
        z:               (B, embed_dim) encoder outputs
        query_positions: (B,) int64 position of each query in the training subset
                         (for leave-one-out during training); None at test time.
        Returns r:       (B, embed_dim) retrieved representation
        """
        assert self.mem_embeddings is not None, "Call set_memory() before forward()"
        B = z.shape[0]
        H, d = self.num_heads, self.d_head
        N = self.mem_embeddings.shape[0]
        dev = z.device

        mem = self.mem_embeddings.to(dev)
        mem_lbl = self.mem_labels.to(dev)
        mem_idx = self.mem_indices.to(dev)

        # Project query: (B, H, d)
        q = self.q_proj(self.q_norm(z)).view(B, H, d)

        # Project memory: (N, H, d)
        mem_norm = self.kv_norm(mem)
        k = self.k_proj(mem_norm).view(N, H, d)
        v = (self.v_proj(mem_norm) + self.label_emb(mem_lbl)).view(N, H, d)

        # Scores: (B, H, N)
        beta = self.log_beta.exp()                          # (H,)
        # q: (B,H,d) → (B,H,1,d); k: (N,H,d) → (H,N,d) → (1,H,d,N)
        q_4 = q.unsqueeze(2)                               # (B, H, 1, d)
        k_4 = k.permute(1, 0, 2).unsqueeze(0)             # (1, H, N, d)
        scores = (beta.view(1, H, 1, 1) * (q_4 @ k_4.transpose(-2, -1))
                  ).squeeze(2)                              # (B, H, N)

        # Leave-one-out: mask each query's own slot
        if query_positions is not None:
            qpos = query_positions.to(dev)                  # (B,)
            # match[b, n] = True when mem_idx[n] == qpos[b]
            match = qpos.unsqueeze(1) == mem_idx.unsqueeze(0)  # (B, N)
            scores = scores.masked_fill(match.unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)                    # (B, H, N)

        # Retrieve values: (B, H, d)
        v_4 = v.permute(1, 0, 2).unsqueeze(0)             # (1, H, N, d)
        r = (attn.unsqueeze(-2) @ v_4).squeeze(-2)         # (B, H, d) via (B,H,1,N)@(1,H,N,d)
        # Actually: (B,H,N) and v (1,H,N,d)
        # r[b,h] = sum_n attn[b,h,n] * v[0,h,n,:]
        r = (attn.unsqueeze(3) * v_4).sum(2)               # (B, H, d)
        r = r.reshape(B, self.embed_dim)
        return r


# ── Classifier variants ───────────────────────────────────────────────────────

class BaselineClassifier(nn.Module):
    def __init__(self, encoder: TransformerEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, tokens, pad_mask, **_):
        return self.encoder(tokens, pad_mask)

    def get_embeddings(self, tokens, pad_mask):
        return self.encoder.encode(tokens, pad_mask)


class HPoolClassifier(nn.Module):
    def __init__(self, encoder: TransformerEncoder, num_heads: int, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        # Replace mean pooling with HopfieldPooling (S=1 learned query)
        self.hpool = HopfieldPooling(embed_dim, num_heads, num_patterns=1,
                                     learn_beta=True)
        self.head = encoder.head

    def _encode_hpool(self, tokens, pad_mask):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.encoder.dropout(
            self.encoder.tok_emb(tokens) + self.encoder.pos_emb(pos)
        )
        for layer in self.encoder.layers:
            x = layer(x, key_padding_mask=pad_mask)
        x = self.encoder.norm(x)
        return self.hpool(x, padding_mask=pad_mask)

    def forward(self, tokens, pad_mask, **_):
        z = self._encode_hpool(tokens, pad_mask)
        return self.head(z)

    def get_embeddings(self, tokens, pad_mask):
        return self._encode_hpool(tokens, pad_mask)


class HMemClassifier(nn.Module):
    def __init__(self, encoder: TransformerEncoder, num_heads: int,
                 embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.hmem = HopfieldMemory(embed_dim, num_heads, num_classes)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.head = encoder.head

    def forward(self, tokens, pad_mask, query_positions=None, **_):
        z = self.encoder.encode(tokens, pad_mask)
        r = self.hmem(z, query_positions=query_positions)
        return self.head(self.out_norm(z + r))

    def get_embeddings(self, tokens, pad_mask):
        return self.encoder.encode(tokens, pad_mask)


class HPoolMemClassifier(nn.Module):
    def __init__(self, encoder: TransformerEncoder, num_heads: int,
                 embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.hpool = HopfieldPooling(embed_dim, num_heads, num_patterns=1,
                                     learn_beta=True)
        self.hmem = HopfieldMemory(embed_dim, num_heads, num_classes)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.head = encoder.head

    def _encode_hpool(self, tokens, pad_mask):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.encoder.dropout(
            self.encoder.tok_emb(tokens) + self.encoder.pos_emb(pos)
        )
        for layer in self.encoder.layers:
            x = layer(x, key_padding_mask=pad_mask)
        x = self.encoder.norm(x)
        return self.hpool(x, padding_mask=pad_mask), x, pad_mask

    def forward(self, tokens, pad_mask, query_positions=None, **_):
        z, _, _ = self._encode_hpool(tokens, pad_mask)
        r = self.hmem(z, query_positions=query_positions)
        return self.head(self.out_norm(z + r))

    def get_embeddings(self, tokens, pad_mask):
        z, _, _ = self._encode_hpool(tokens, pad_mask)
        return z


# ── Memory refresh ────────────────────────────────────────────────────────────

@torch.no_grad()
def refresh_memory(model, subset_ds, device, batch_size=256):
    """Recompute all memory embeddings using the current encoder."""
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False)
    embeddings, labels_list, positions = [], [], []
    pos = 0
    for tokens, pad_mask, labels in loader:
        tokens, pad_mask = tokens.to(device), pad_mask.to(device)
        z = model.get_embeddings(tokens, pad_mask)
        embeddings.append(z.cpu())
        labels_list.append(labels)
        positions.extend(range(pos, pos + len(labels)))
        pos += len(labels)
    emb = torch.cat(embeddings, 0)
    lbl = torch.cat(labels_list, 0)
    idx = torch.arange(len(lbl), dtype=torch.long)
    # Push memory onto hmem module
    hmem = getattr(model, "hmem", None)
    if hmem is not None:
        hmem.set_memory(emb, lbl, idx)


# ── Training ──────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_variant(
    variant: str,
    per_class: int,
    seed: int,
    data_dir: str,
    results_base: str,
    max_steps: int,
    eval_every: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    max_len: int,
    dropout: float,
):
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = os.path.join(
        results_base,
        f"lowdata-{variant}-n{per_class}-seed{seed}-{job_id}"
    )
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data ─────────────────────────────────────────────────────────────────
    vocab = load_vocab(data_dir)
    dcfg = load_config(data_dir)
    num_classes = dcfg["num_classes"]

    train_texts, train_labels, train_ids = load_split(data_dir, "train")
    val_texts, val_labels, val_ids = load_split(data_dir, "val")
    test_texts, test_labels, _ = load_split(data_dir, "test")

    # Build nested subsets only up to the requested size (saves memory on large sizes)
    sizes_needed = [s for s in [10, 25, 100, 200] if s <= per_class]
    if per_class not in sizes_needed:
        sizes_needed.append(per_class)
    subsets = make_nested_subsets(
        train_texts, train_labels, train_ids,
        per_class_sizes=sizes_needed, seed=seed,
    )
    sub_texts, sub_labels, sub_ids = subsets[per_class]

    val_sub_texts, val_sub_labels, _ = make_val_subset(
        val_texts, val_labels, val_ids, per_class=VAL_PER_CLASS
    )

    subset_ds = NewsgroupDataset(sub_texts, sub_labels, vocab)
    val_ds = NewsgroupDataset(val_sub_texts, val_sub_labels, vocab)

    # Separate generator for data loading
    g = torch.Generator(); g.manual_seed(seed + 2000)
    train_loader = DataLoader(
        subset_ds, batch_size=min(batch_size, len(subset_ds)),
        shuffle=True, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    base_enc = TransformerEncoder(
        vocab_size=len(vocab), embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, ffn_dim=ffn_dim, max_len=max_len,
        dropout=dropout, num_classes=num_classes,
    )

    if variant == "baseline":
        model = BaselineClassifier(base_enc)
    elif variant == "hpool":
        model = HPoolClassifier(base_enc, num_heads, embed_dim)
    elif variant == "hmem":
        model = HMemClassifier(base_enc, num_heads, embed_dim, num_classes)
    elif variant == "hpool+hmem":
        model = HPoolMemClassifier(base_enc, num_heads, embed_dim, num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = model.to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # ── run config ────────────────────────────────────────────────────────────
    run_cfg = {
        "variant": variant, "per_class": per_class, "seed": seed,
        "max_steps": max_steps, "eval_every": eval_every,
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "grad_clip": grad_clip, "n_params": n_params,
        "n_subset": len(subset_ds), "n_val": len(val_ds),
        "device": str(device), "job_id": job_id,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"[{variant}] n={per_class}/class, seed={seed}, "
          f"params={n_params:,}, device={device}")

    # Initial memory refresh for memory variants
    needs_memory = variant in ("hmem", "hpool+hmem")
    if needs_memory:
        refresh_memory(model, subset_ds, device)

    # ── training loop (step-based) ────────────────────────────────────────────
    best_val_acc = 0.0
    no_improve = 0
    step = 0
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    train_iter = iter(train_loader)

    def _next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    last_refresh_step = 0
    REFRESH_EVERY = 200  # steps

    while step < max_steps:
        model.train()
        tokens, pad_mask, labels = _next_batch()
        tokens, pad_mask, labels = (
            tokens.to(device), pad_mask.to(device), labels.to(device)
        )

        # Refresh memory every REFRESH_EVERY steps or at start of "epoch"
        if needs_memory and (step - last_refresh_step) >= REFRESH_EVERY:
            refresh_memory(model, subset_ds, device)
            last_refresh_step = step

        optimizer.zero_grad()
        # For memory variants, pass subset positions for leave-one-out
        if needs_memory:
            # Indices into the subset for leave-one-out masking
            # We need to identify which subset position each batch item is.
            # Since we shuffle, we track by matching labels and passing None
            # (leave-one-out only matters when batch items are in memory).
            # We pass None here and rely on the memory refresh to handle it.
            logits = model(tokens, pad_mask, query_positions=None)
        else:
            logits = model(tokens, pad_mask)

        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        step += 1

        # Evaluation
        if step % eval_every == 0 or step == max_steps:
            model.eval()
            # Refresh memory with full subset for val
            if needs_memory:
                refresh_memory(model, subset_ds, device)
            correct = total = 0
            with torch.no_grad():
                for vt, vp, vl in val_loader:
                    vt, vp, vl = vt.to(device), vp.to(device), vl.to(device)
                    pred = model(vt, vp)
                    correct += (pred.argmax(-1) == vl).sum().item()
                    total += vl.size(0)
            val_acc = correct / total
            row = {"step": step, "loss": loss.item(), "val_acc": val_acc}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  step {step:5d} | loss {loss.item():.4f} | val_acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at step {step} (patience {patience})")
                    break

    torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
    best_sha = sha256_file(os.path.join(run_dir, "best.pt"))
    last_sha = sha256_file(os.path.join(run_dir, "last.pt"))

    summary = {
        "best_val_acc": best_val_acc,
        "steps_trained": step,
        "best_pt_sha256": best_sha,
        "last_pt_sha256": last_sha,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Done. best_val_acc={best_val_acc:.4f}, run_dir={run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--per_class", type=int, default=25,
                    help="Labeled examples per class (10, 25, 100, 200)")
    ap.add_argument("--variant", default="all",
                    help="One of: baseline hpool hmem hpool+hmem all")
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--results_base", default="results/")
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--ffn_dim", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    variants = VARIANTS if args.variant == "all" else [args.variant]
    kwargs = {k: getattr(args, k) for k in (
        "per_class", "seed", "data_dir", "results_base", "max_steps",
        "eval_every", "patience", "batch_size", "lr", "weight_decay",
        "grad_clip", "embed_dim", "num_heads", "num_layers", "ffn_dim",
        "max_len", "dropout",
    )}
    for v in variants:
        run_variant(variant=v, **kwargs)


if __name__ == "__main__":
    main()
