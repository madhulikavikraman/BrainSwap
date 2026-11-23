"""
Banking77 data pipeline.

Downloads Banking77 via HuggingFace datasets, tokenizes with the same
regex as the 20 Newsgroups pipeline, builds a 20k vocab from training
texts only, and saves reproducible train/val/test splits.

Usage:
    python src/data_pipeline_banking77.py --data_dir data/banking77/

Outputs in data_dir/:
    config.json            vocab size, split seed, class list
    vocab.json             word -> int (pad=0, unk=1, words start at 2)
    train_texts.pkl        list[str]  - 90% of official train
    train_labels.pkl       list[int]
    train_ids.pkl          list[int]
    val_texts.pkl          list[str]  - 10% of official train
    val_labels.pkl         list[int]
    val_ids.pkl            list[int]
    test_texts.pkl         list[str]  - official test split
    test_labels.pkl        list[int]
    test_ids.pkl           list[int]
"""
import argparse
import json
import os
import pickle
import random
import re
from collections import Counter
from typing import List, Tuple

SPLIT_SEED = 544
VAL_FRACTION = 0.10
VOCAB_SIZE = 20_000
MAX_LEN = 128
PAD_ID = 0
UNK_ID = 1

_TOKEN_RE = re.compile(r"[a-z]+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: List[str]) -> dict:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_ids(texts: List[str], vocab: dict) -> List[List[int]]:
    result = []
    for text in texts:
        toks = tokenize(text)[:MAX_LEN]
        ids = [vocab.get(t, UNK_ID) for t in toks]
        if not ids:
            ids = [UNK_ID]
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
        result.append(ids)
    return result


def _load_banking77() -> Tuple[List[str], List[int], List[str], List[int]]:
    from datasets import load_dataset
    ds = load_dataset("banking77")
    train_texts  = [ex["text"]  for ex in ds["train"]]
    train_labels = [ex["label"] for ex in ds["train"]]
    test_texts   = [ex["text"]  for ex in ds["test"]]
    test_labels  = [ex["label"] for ex in ds["test"]]
    return train_texts, train_labels, test_texts, test_labels


def prepare(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)

    print("Loading Banking77 from HuggingFace ...")
    all_train_texts, all_train_labels, test_texts, test_labels = _load_banking77()

    # get class names from HuggingFace feature info
    from datasets import load_dataset
    ds = load_dataset("banking77")
    class_names: List[str] = ds["train"].features["label"].names
    num_classes = len(class_names)  # 77

    # stratified 90/10 train/val split
    rng = random.Random(SPLIT_SEED)
    by_class: dict = {}
    for i, lbl in enumerate(all_train_labels):
        by_class.setdefault(lbl, []).append(i)

    val_ids, train_ids = [], []
    for lbl in sorted(by_class):
        idxs = by_class[lbl]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * VAL_FRACTION))
        val_ids.extend(idxs[:n_val])
        train_ids.extend(idxs[n_val:])

    train_ids.sort(); val_ids.sort()
    train_texts = [all_train_texts[i] for i in train_ids]
    train_labels_split = [all_train_labels[i] for i in train_ids]
    val_texts   = [all_train_texts[i] for i in val_ids]
    val_labels  = [all_train_labels[i] for i in val_ids]
    test_ids    = list(range(len(test_texts)))

    print("Building vocabulary ...")
    vocab = build_vocab(train_texts)

    for name, obj in [
        ("train_texts", train_texts), ("train_labels", train_labels_split),
        ("train_ids", train_ids),
        ("val_texts", val_texts), ("val_labels", val_labels),
        ("val_ids", val_ids),
        ("test_texts", test_texts), ("test_labels", test_labels),
        ("test_ids", test_ids),
    ]:
        with open(os.path.join(data_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)

    with open(os.path.join(data_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    config = {
        "dataset": "banking77",
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "vocab_size": len(vocab),
        "max_len": MAX_LEN,
        "num_classes": num_classes,
        "classes": class_names,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
    }
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done. Train {len(train_texts)}, val {len(val_texts)}, test {len(test_texts)}")
    print(f"Classes: {num_classes}  Vocab size: {len(vocab)}")


import torch


class Banking77Dataset(torch.utils.data.Dataset):
    """Token-ID dataset; each item is (tokens, pad_mask, label)."""

    def __init__(self, texts: List[str], labels: List[int], vocab: dict):
        self.ids = texts_to_ids(texts, vocab)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.ids[idx], dtype=torch.long)
        pad_mask = tokens == PAD_ID
        return tokens, pad_mask, torch.tensor(self.labels[idx], dtype=torch.long)


def load_split(data_dir: str, split: str):
    def _load(name):
        with open(os.path.join(data_dir, f"{name}.pkl"), "rb") as f:
            return pickle.load(f)
    return _load(f"{split}_texts"), _load(f"{split}_labels"), _load(f"{split}_ids")


def load_vocab(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "vocab.json")) as f:
        return json.load(f)


def load_config(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "config.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/banking77/")
    args = ap.parse_args()
    prepare(args.data_dir)
