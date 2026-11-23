"""
Stratified low-data subset sampling for Track A.

Subsets are NESTED within a seed: the 10-per-class subset is always a subset
of the 25-per-class subset, which is a subset of the 100-per-class, etc.
All variants at the same seed see exactly the same articles.

Usage (as a library):
    from lowdata_subsets import make_nested_subsets
    subsets = make_nested_subsets(train_texts, train_labels, train_ids,
                                  per_class_sizes=[10, 25, 100, 200], seed=27)
    # subsets[25] == (texts, labels, orig_ids)
"""
import random
from typing import Dict, List, Tuple

PER_CLASS_SIZES = [10, 25, 100, 200]


def make_nested_subsets(
    texts: List[str],
    labels: List[int],
    orig_ids: List[int],
    per_class_sizes: List[int] = PER_CLASS_SIZES,
    seed: int = 27,
) -> Dict[int, Tuple[List[str], List[int], List[int]]]:
    """
    Return a dict mapping per_class_size → (texts, labels, orig_ids).
    Subsets are nested: smaller ⊂ larger for the same seed.
    """
    rng = random.Random(seed)

    # Group indices by class
    by_class: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        by_class.setdefault(lbl, []).append(i)

    # Shuffle once per class (order determines which items go into each subset)
    for lbl in by_class:
        rng.shuffle(by_class[lbl])

    max_size = max(per_class_sizes)

    # Validate class sizes
    for lbl, idxs in by_class.items():
        if len(idxs) < max_size:
            raise ValueError(
                f"Class {lbl} has only {len(idxs)} training examples, "
                f"but max per_class_size is {max_size}."
            )

    subsets: Dict[int, Tuple[List[str], List[int], List[int]]] = {}
    for n in sorted(per_class_sizes):
        sel_texts, sel_labels, sel_ids = [], [], []
        for lbl in sorted(by_class):
            for i in by_class[lbl][:n]:
                sel_texts.append(texts[i])
                sel_labels.append(labels[i])
                sel_ids.append(orig_ids[i])
        subsets[n] = (sel_texts, sel_labels, sel_ids)

    return subsets


def make_val_subset(
    val_texts: List[str],
    val_labels: List[int],
    val_ids: List[int],
    per_class: int = 200,
    seed: int = 544,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Fixed validation subset: per_class examples per class, same for all sizes/seeds.
    Drawn from the prepared val split (never from train or test).
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, lbl in enumerate(val_labels):
        by_class.setdefault(lbl, []).append(i)

    sel_texts, sel_labels, sel_ids = [], [], []
    for lbl in sorted(by_class):
        idxs = by_class[lbl]
        rng.shuffle(idxs)
        for i in idxs[:per_class]:
            sel_texts.append(val_texts[i])
            sel_labels.append(val_labels[i])
            sel_ids.append(val_ids[i])

    return sel_texts, sel_labels, sel_ids


if __name__ == "__main__":
    # Smoke test: print subset sizes and verify nesting
    import sys
    sys.path.insert(0, "src")
    from data_pipeline import load_split, load_config

    DATA_DIR = "data/"
    try:
        texts, labels, ids = load_split(DATA_DIR, "train")
        cfg = load_config(DATA_DIR)
    except FileNotFoundError:
        # Synthetic test data
        import string
        n = 1000
        nc = 20
        texts = [f"article {i}" for i in range(n)]
        labels = [i % nc for i in range(n)]
        ids = list(range(n))

    subsets = make_nested_subsets(texts, labels, ids)
    for n in sorted(subsets):
        t, l, _ = subsets[n]
        from collections import Counter
        cnt = Counter(l)
        print(f"  n={n:4d}: total={len(t)}, per-class min/max = "
              f"{min(cnt.values())}/{max(cnt.values())}")

    # Verify nesting
    s10_ids = set(subsets[10][2])
    s25_ids = set(subsets[25][2])
    s100_ids = set(subsets[100][2])
    s200_ids = set(subsets[200][2])
    assert s10_ids <= s25_ids <= s100_ids <= s200_ids, "Subsets are not nested!"
    print("Nesting OK.")
