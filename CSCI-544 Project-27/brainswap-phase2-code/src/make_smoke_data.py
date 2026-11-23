"""
Creates minimal synthetic data in data/ matching the exact format of data_pipeline.py.
Used for local CPU smoke tests only. Not uploaded to CARC.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import NEWSGROUPS, PAD_ID, UNK_ID, MAX_LEN, VOCAB_SIZE

DATA_DIR = "data/"
NUM_CLASSES = 20
N_TRAIN = 200    # 10 per class
N_VAL = 40       # 2 per class
N_TEST = 80      # 4 per class
VOCAB_SZ = 1000  # small for speed

os.makedirs(DATA_DIR, exist_ok=True)

import random
rng = random.Random(0)

def _rand_text(i):
    words = [f"word{rng.randint(0, VOCAB_SZ-1)}" for _ in range(rng.randint(20, 60))]
    return " ".join(words)

def _make_split(n, nc, prefix):
    texts, labels, ids = [], [], []
    for c in range(nc):
        for j in range(n // nc):
            idx = c * (n // nc) + j
            texts.append(_rand_text(idx))
            labels.append(c)
            ids.append(len(ids))
    return texts, labels, ids

train_t, train_l, train_i = _make_split(N_TRAIN, NUM_CLASSES, "train")
val_t, val_l, val_i = _make_split(N_VAL, NUM_CLASSES, "val")
test_t, test_l, test_i = _make_split(N_TEST, NUM_CLASSES, "test")

# Build tiny vocab
vocab = {"<pad>": 0, "<unk>": 1}
for w in [f"word{i}" for i in range(VOCAB_SZ - 2)]:
    vocab[w] = len(vocab)

for name, obj in [
    ("train_texts", train_t), ("train_labels", train_l), ("train_ids", train_i),
    ("val_texts", val_t),   ("val_labels", val_l),   ("val_ids", val_i),
    ("test_texts", test_t), ("test_labels", test_l), ("test_ids", test_i),
]:
    with open(os.path.join(DATA_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(obj, f)

with open(os.path.join(DATA_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f)

config = {
    "dataset": "smoke-synthetic",
    "split_seed": 544, "val_fraction": 0.1,
    "vocab_size": len(vocab), "max_len": MAX_LEN,
    "num_classes": NUM_CLASSES, "classes": NEWSGROUPS,
    "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
    "data_sha256": "smoke-test-no-sha",
}
with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"Smoke data written to {DATA_DIR}")
print(f"  train={len(train_t)}, val={len(val_t)}, test={len(test_t)}, vocab={len(vocab)}")
