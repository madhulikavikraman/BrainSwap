"""
Data pipeline for 20 Newsgroups (bydate split).

Downloads 20news-bydate.tar.gz, strips email headers, tokenizes with the
same regex as the BrainSwap AG-News pipeline, builds a 20k vocabulary from
training texts only, and creates reproducible train/val/test splits.

Usage:
    python src/data_pipeline.py --data_dir data/

Outputs in data_dir/:
    20news-bydate.tar.gz   raw download
    config.json            vocab size, split seed, class list, data signature
    vocab.json             word → int (pad=0, unk=1, words start at 2)
    train_texts.pkl        list[str]  – 90% of bydate-train
    train_labels.pkl       list[int]
    train_ids.pkl          list[int]  – row index in original bydate-train
    val_texts.pkl          list[str]  – 10% of bydate-train
    val_labels.pkl         list[int]
    val_ids.pkl            list[int]
    test_texts.pkl         list[str]  – bydate-test
    test_labels.pkl        list[int]
    test_ids.pkl           list[int]
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import re
import tarfile
import urllib.request
from collections import Counter
from typing import List, Tuple

DATASET_URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
SPLIT_SEED = 544     # same as AG-News pipeline in the brief
VAL_FRACTION = 0.10  # 10% of training set goes to validation
VOCAB_SIZE = 20_000  # same as AG-News
MAX_LEN = 128        # same as AG-News
PAD_ID = 0
UNK_ID = 1

NEWSGROUPS = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(NEWSGROUPS)}
NUM_CLASSES = len(NEWSGROUPS)  # 20

_TOKEN_RE = re.compile(r"[a-z]+|[^\w\s]")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _strip_header(text: str) -> str:
    """Remove the email header (everything up to and including the first blank line)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "":
            return "\n".join(lines[i + 1 :])
    return text  # no blank line found; return as-is


def _read_newsgroup_dir(root: str) -> Tuple[List[str], List[int]]:
    """Read all documents from a bydate-{train,test}/ tree."""
    texts, labels = [], []
    for group in sorted(os.listdir(root)):
        group_path = os.path.join(root, group)
        if not os.path.isdir(group_path):
            continue
        label = CLASS_TO_IDX.get(group)
        if label is None:
            continue
        for fname in sorted(os.listdir(group_path)):
            fpath = os.path.join(group_path, fname)
            try:
                with open(fpath, "r", encoding="latin-1") as f:
                    raw = f.read()
            except OSError:
                continue
            texts.append(_strip_header(raw))
            labels.append(label)
    return texts, labels


def build_vocab(texts: List[str]) -> dict:
    """Return {word: id} from the top VOCAB_SIZE-2 most frequent tokens."""
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_ids(texts: List[str], vocab: dict) -> List[List[int]]:
    """Tokenize and map to IDs, truncating/padding to MAX_LEN."""
    result = []
    for text in texts:
        toks = tokenize(text)[:MAX_LEN]
        ids = [vocab.get(t, UNK_ID) for t in toks]
        if len(ids) == 0:
            ids = [UNK_ID]
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
        result.append(ids)
    return result


def prepare(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "20news-bydate.tar.gz")

    # Download
    if not os.path.exists(tar_path):
        print(f"Downloading 20news-bydate.tar.gz …")
        urllib.request.urlretrieve(DATASET_URL, tar_path)
    print(f"SHA-256: {_sha256(tar_path)}")

    # Extract
    extract_dir = os.path.join(data_dir, "20news-bydate")
    if not os.path.exists(extract_dir):
        print("Extracting …")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        # tarball extracts to 20news-bydate-train/ and 20news-bydate-test/
        os.rename(os.path.join(data_dir, "20news-bydate-train"),
                  os.path.join(data_dir, "20news-bydate-train_raw"))

    train_raw_dir = os.path.join(data_dir, "20news-bydate-train_raw")
    test_raw_dir = os.path.join(data_dir, "20news-bydate-test")

    print("Reading training documents …")
    all_train_texts, all_train_labels = _read_newsgroup_dir(train_raw_dir)
    print("Reading test documents …")
    test_texts, test_labels = _read_newsgroup_dir(test_raw_dir)

    # Stratified train/val split
    rng = random.Random(SPLIT_SEED)
    by_class = {}
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
    train_labels = [all_train_labels[i] for i in train_ids]
    val_texts = [all_train_texts[i] for i in val_ids]
    val_labels = [all_train_labels[i] for i in val_ids]
    test_ids = list(range(len(test_texts)))

    # Vocabulary from training texts only
    print("Building vocabulary …")
    vocab = build_vocab(train_texts)

    # Save everything
    for name, obj in [
        ("train_texts", train_texts), ("train_labels", train_labels),
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
        "dataset": "20news-bydate",
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "vocab_size": len(vocab),
        "max_len": MAX_LEN,
        "num_classes": NUM_CLASSES,
        "classes": NEWSGROUPS,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "data_sha256": _sha256(tar_path),
    }
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done. Train {len(train_texts)}, val {len(val_texts)}, test {len(test_texts)}")
    print(f"Vocab size: {len(vocab)}")


# ---------------------------------------------------------------------------
# Dataset class used by all training scripts
# ---------------------------------------------------------------------------

import torch


class NewsgroupDataset(torch.utils.data.Dataset):
    """Token-ID dataset; each item is (tokens, pad_mask, label)."""

    def __init__(self, texts: List[str], labels: List[int], vocab: dict):
        self.ids = texts_to_ids(texts, vocab)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.ids[idx], dtype=torch.long)
        pad_mask = tokens == PAD_ID   # True where padding
        return tokens, pad_mask, torch.tensor(self.labels[idx], dtype=torch.long)


def load_split(data_dir: str, split: str):
    """Return (texts, labels, ids) for split in {'train','val','test'}."""
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
    # Deferred import so torch is not required just to run the downloader
    import torch  # noqa: F401 – needed by NewsgroupDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/")
    args = ap.parse_args()
    prepare(args.data_dir)
