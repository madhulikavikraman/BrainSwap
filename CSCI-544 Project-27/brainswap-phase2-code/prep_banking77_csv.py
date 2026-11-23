"""Banking77 without HuggingFace: download PolyAI's original CSVs, then reuse Siddu's prepare()."""
import csv, io, sys, types, urllib.request
sys.path.insert(0, "src")
import datasets
import data_pipeline_banking77 as b77

URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"
def read(split):
    text = urllib.request.urlopen(URL.format(split), timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    return [r["text"] for r in rows], [r["category"] for r in rows]

train_x, train_c = read("train")
test_x, test_c = read("test")
names = sorted(set(train_c))
idx = {n: i for i, n in enumerate(names)}
assert len(names) == 77 and len(train_x) == 10003 and len(test_x) == 3080, "unexpected download"

b77._load_banking77 = lambda: (train_x, [idx[c] for c in train_c], test_x, [idx[c] for c in test_c])
fake = {"train": types.SimpleNamespace(features={"label": types.SimpleNamespace(names=names)})}
datasets.load_dataset = lambda *a, **k: fake
b77.prepare(sys.argv[1] if len(sys.argv) > 1 else "data/banking77/")
