"""Fix: hmem's leave-one-out mask was never applied during training.

train_lowdata.py calls model(..., query_positions=None), so every training example can
retrieve its OWN embedding + label from memory. The model learns to copy that label,
and the shortcut disappears at validation time. This patch passes real subset positions.
Safe to run twice; aborts without writing if the file doesn't look as expected.
"""
import shutil, sys

path = sys.argv[1]
src = open(path).read()
if "class IndexedDataset" in src:
    print("already patched"); sys.exit(0)

edits = [
    ("VARIANTS = [",
     'class IndexedDataset(torch.utils.data.Dataset):\n'
     '    """Returns (tokens, pad_mask, label, position) so hmem can mask each item\'s own slot."""\n'
     '    def __init__(self, ds):\n        self.ds = ds\n'
     '    def __len__(self):\n        return len(self.ds)\n'
     '    def __getitem__(self, i):\n        return (*self.ds[i], i)\n\n\n'
     'VARIANTS = ['),
    ("        subset_ds, batch_size=min(batch_size, len(subset_ds)),",
     "        IndexedDataset(subset_ds), batch_size=min(batch_size, len(subset_ds)),"),
    ("tokens, pad_mask, labels = _next_batch()",
     "tokens, pad_mask, labels, batch_idx = _next_batch()"),
    ("logits = model(tokens, pad_mask, query_positions=None)",
     "logits = model(tokens, pad_mask, query_positions=batch_idx.to(device))"),
]
for old, _ in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"ABORT: expected 1 match, found {n}: {old!r}. File unchanged; send it to Claude.")
for old, new in edits:
    src = src.replace(old, new)
shutil.copy(path, path + ".orig")
open(path, "w").write(src)
print(f"patched {path} (backup: {path}.orig)")
