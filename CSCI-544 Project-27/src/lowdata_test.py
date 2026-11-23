"""
Test-set evaluation for Track A (run once, after variant selection).

Loads best.pt checkpoints from all variant run directories for a given
(seed, per_class_size) combination and evaluates on the full test set.

Usage:
    python src/lowdata_test.py \\
        --results_base results/ --data_dir data/ \\
        --seeds 27 28 29 --per_class_sizes 10 25 100 200

Outputs:
    results/lowdata-test-results.json
    (appended / updated with new per_class / seed combinations)
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import NewsgroupDataset, load_split, load_vocab, load_config
from model import TransformerEncoder
from train_lowdata import (
    BaselineClassifier, HPoolClassifier, HMemClassifier, HPoolMemClassifier,
    refresh_memory,
)
from lowdata_subsets import make_nested_subsets

VARIANTS = ["baseline", "hpool", "hmem", "hpool+hmem"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_run_dir(results_base, variant, per_class, seed):
    pattern = os.path.join(
        results_base, f"lowdata-{variant}-n{per_class}-seed{seed}-*"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]  # latest job


def _build_model(variant, encoder, num_heads, embed_dim, num_classes):
    if variant == "baseline":
        return BaselineClassifier(encoder)
    if variant == "hpool":
        return HPoolClassifier(encoder, num_heads, embed_dim)
    if variant == "hmem":
        return HMemClassifier(encoder, num_heads, embed_dim, num_classes)
    if variant == "hpool+hmem":
        return HPoolMemClassifier(encoder, num_heads, embed_dim, num_classes)
    raise ValueError(variant)


def evaluate_variant(
    variant, per_class, seed, results_base, data_dir, device
):
    run_dir = _find_run_dir(results_base, variant, per_class, seed)
    if run_dir is None:
        print(f"  MISSING: {variant} n={per_class} seed={seed}")
        return None

    vocab = load_vocab(data_dir)
    dcfg = load_config(data_dir)
    num_classes = dcfg["num_classes"]
    embed_dim = 256; num_heads = 4

    train_texts, train_labels, train_ids = load_split(data_dir, "train")
    test_texts, test_labels, _ = load_split(data_dir, "test")

    sizes_needed = [s for s in [10, 25, 100, 200] if s <= per_class]
    if per_class not in sizes_needed:
        sizes_needed.append(per_class)
    subsets = make_nested_subsets(
        train_texts, train_labels, train_ids,
        per_class_sizes=sizes_needed, seed=seed,
    )
    sub_texts, sub_labels, _ = subsets[per_class]

    torch.manual_seed(seed)
    base_enc = TransformerEncoder(
        vocab_size=len(vocab), embed_dim=embed_dim, num_heads=num_heads,
        num_layers=4, ffn_dim=1024, max_len=128, dropout=0.1,
        num_classes=num_classes,
    )
    model = _build_model(variant, base_enc, num_heads, embed_dim, num_classes)
    model = model.to(device)

    ckpt_path = os.path.join(run_dir, "best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Refresh memory at test time (no leave-one-out)
    if variant in ("hmem", "hpool+hmem"):
        subset_ds = NewsgroupDataset(sub_texts, sub_labels, vocab)
        refresh_memory(model, subset_ds, device)

    test_ds = NewsgroupDataset(test_texts, test_labels, vocab)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    correct = total = 0
    with torch.no_grad():
        for tokens, pad_mask, labels in test_loader:
            tokens, pad_mask, labels = (
                tokens.to(device), pad_mask.to(device), labels.to(device)
            )
            logits = model(tokens, pad_mask)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return {
        "acc": acc,
        "correct": correct,
        "total": total,
        "ckpt_sha256": sha256_file(ckpt_path),
        "run_dir": run_dir,
    }


def collect_results(seeds, per_class_sizes, results_base, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np

    output = {}
    for pc in per_class_sizes:
        output[pc] = {}
        for variant in VARIANTS:
            accs = []
            for seed in seeds:
                res = evaluate_variant(variant, pc, seed, results_base, data_dir, device)
                if res:
                    print(f"  {variant} n={pc} seed={seed}: acc={res['acc']:.4f}")
                    accs.append(res["acc"])
            if accs:
                output[pc][variant] = {
                    "per_seed_acc": accs,
                    "mean_acc": float(np.mean(accs)),
                    "std_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                }
                # Paired differences vs baseline
                if variant != "baseline" and "baseline" in output[pc]:
                    bl_accs = output[pc]["baseline"].get("per_seed_acc", [])
                    if len(bl_accs) == len(accs):
                        diffs = [a - b for a, b in zip(accs, bl_accs)]
                        output[pc][variant]["diff_vs_baseline"] = diffs
                        output[pc][variant]["mean_diff"] = float(np.mean(diffs))

    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", default="results/")
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--seeds", nargs="+", type=int, default=[27, 28, 29])
    ap.add_argument("--per_class_sizes", nargs="+", type=int,
                    default=[10, 25, 100, 200])
    ap.add_argument("--output", default="results/lowdata-test-results.json")
    args = ap.parse_args()

    results = collect_results(
        args.seeds, args.per_class_sizes, args.results_base, args.data_dir
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'per_class':>10}", end="")
    for v in VARIANTS:
        print(f"  {v:>14}", end="")
    print()
    for pc in args.per_class_sizes:
        print(f"{pc:>10}", end="")
        for v in VARIANTS:
            rec = results.get(pc, {}).get(v, {})
            if rec:
                m = rec["mean_acc"] * 100
                s = rec["std_acc"] * 100
                print(f"  {m:5.2f}±{s:4.2f}    ", end="")
            else:
                print(f"  {'n/a':>14}", end="")
        print()


if __name__ == "__main__":
    main()
