"""
Prepares real 20 Newsgroups data using sklearn (fast, no tarball download).
Saves in identical format to data_pipeline.py so all training scripts work unchanged.
Run: python src/prep_real_data.py --data_dir data/
"""
import argparse, json, os, pickle, random
from collections import Counter
from typing import List
import re

from sklearn.datasets import fetch_20newsgroups

SPLIT_SEED   = 544
VAL_FRACTION = 0.10
VOCAB_SIZE   = 20_000
MAX_LEN      = 128
PAD_ID       = 0
UNK_ID       = 1

_TOKEN_RE = re.compile(r"[a-z]+|[^\w\s]")

NEWSGROUPS = [
    "alt.atheism","comp.graphics","comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x",
    "misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball",
    "rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space",
    "soc.religion.christian","talk.politics.guns","talk.politics.mideast",
    "talk.politics.misc","talk.religion.misc",
]

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def _strip_header(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "":
            return "\n".join(lines[i+1:])
    return text

def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab

def texts_to_ids(texts, vocab):
    result = []
    for text in texts:
        toks = tokenize(text)[:MAX_LEN]
        ids  = [vocab.get(t, UNK_ID) for t in toks]
        if not ids:
            ids = [UNK_ID]
        ids += [PAD_ID] * (MAX_LEN - len(ids))
        result.append(ids)
    return result

def prepare(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    print("Fetching 20 Newsgroups via sklearn (bydate split)...")
    raw_train = fetch_20newsgroups(subset="train", remove=("headers","footers","quotes"),
                                   data_home=data_dir)
    raw_test  = fetch_20newsgroups(subset="test",  remove=("headers","footers","quotes"),
                                   data_home=data_dir)

    all_train_texts  = raw_train.data
    all_train_labels = raw_train.target.tolist()
    test_texts       = raw_test.data
    test_labels      = raw_test.target.tolist()

    print(f"  Raw train: {len(all_train_texts)}  test: {len(test_texts)}")

    # Stratified 90/10 train/val split
    rng = random.Random(SPLIT_SEED)
    by_class = {}
    for i, lbl in enumerate(all_train_labels):
        by_class.setdefault(lbl, []).append(i)

    val_ids, train_ids = [], []
    for lbl in sorted(by_class):
        idxs = by_class[lbl][:]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * VAL_FRACTION))
        val_ids.extend(idxs[:n_val])
        train_ids.extend(idxs[n_val:])

    train_ids.sort(); val_ids.sort()
    train_texts  = [all_train_texts[i]  for i in train_ids]
    train_labels = [all_train_labels[i] for i in train_ids]
    val_texts    = [all_train_texts[i]  for i in val_ids]
    val_labels   = [all_train_labels[i] for i in val_ids]
    test_ids     = list(range(len(test_texts)))

    print("Building vocabulary...")
    vocab = build_vocab(train_texts)

    for name, obj in [
        ("train_texts", train_texts), ("train_labels", train_labels), ("train_ids", train_ids),
        ("val_texts",   val_texts),   ("val_labels",   val_labels),   ("val_ids",   val_ids),
        ("test_texts",  test_texts),  ("test_labels",  test_labels),  ("test_ids",  test_ids),
    ]:
        with open(os.path.join(data_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)

    with open(os.path.join(data_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    config = {
        "dataset": "20news-bydate-sklearn",
        "split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
        "vocab_size": len(vocab), "max_len": MAX_LEN,
        "num_classes": 20, "classes": NEWSGROUPS,
        "n_train": len(train_texts), "n_val": len(val_texts), "n_test": len(test_texts),
    }
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done. train={len(train_texts)} val={len(val_texts)} test={len(test_texts)} vocab={len(vocab)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/")
    args = ap.parse_args()
    prepare(args.data_dir)
