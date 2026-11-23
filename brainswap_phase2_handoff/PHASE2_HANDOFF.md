# BrainSwap Phase 2 handoff

Siddarth's first phase is two things: a code change that makes the bank-zero control runnable, and this guide. Everything below assumes the CARC setup in `/project2/xiangren_1987/athota/brainswap-p2` (source `env.sh` first, run from `$ROOT`).

## 1. What changed in the code

The grader asked us to separate temperature from memory in generation. Until now the code could not run the cell that does it: `--bank_size 0` with fixed β always fell back to PyTorch's `nn.MultiheadAttention`, so "bank-zero, fixed β" was really just `attn`.

| File | Change |
| --- | --- |
| `src/generation_models.py` | New `force_hopfield` option. With it, bank 0 + fixed β + 1 step goes through our Hopfield module instead of PyTorch attention. New `hopfield_init` option passed to the module. |
| `src/hopfield_layers.py` | New `init` argument: `"xavier"` (default, what every earlier run used) or `"mha"` (copies PyTorch attention's init scheme). Default behaviour is unchanged. |
| `src/train_generation.py` | New flags `--force_hopfield` and `--hopfield_init {xavier,mha}`. Variant names: `bank0`, `bank0+beta`, `bank64`, `bank64+beta`, `bank0+mhainit`. |
| `src/generation_test.py` | Now finds `bank0`, `bank0+beta` and `bank0+mhainit` runs (before, the bank0+beta control was silently skipped), rebuilds them with the right flags, and reports the mean greedy unknown-token rate. |
| `tests/test_bank_zero.py` | New. 6 checks, listed in section 5. Run with `python tests/test_bank_zero.py`. |
| `gen_controls.sbatch` | New. Runs the two missing 2×2 cells, 3 seeds each. |

Nothing changes for runs already on disk: old `run_config.json` files have no `force_hopfield` key, and the default is off.

The four `src/` files in this folder were rebuilt from the Drive copies of the CARC code, so a few comment lines may differ in spacing from what is on CARC. Behaviour is the same. Back up the CARC originals, then copy these over them. `bank_zero.patch` shows only the real changes, for review.

Checked before handoff: all 6 unit checks pass (PyTorch 2.15, CPU). A smoke run on a tiny made-up corpus trained attn, bank0, bank0+beta, bank64, bank64+beta and bank0+mhainit for 2 epochs each, and `generation_test.py` found and scored all six. The smoke perplexities mean nothing; it only shows the pipeline runs end to end.

### A second difference between `attn` and the Hopfield runs

Besides β, the Hopfield module starts from different weights than PyTorch attention. It applies xavier init to each projection on its own. PyTorch applies one xavier draw to the stacked (768 × 256) q/k/v matrix and the plain `nn.Linear` init to the output projection. At width 256 that makes our q/k/v weights about 1.41× wider and the output projection about 1.73× wider at the start of training (the test prints the measured spreads).

So the midterm's "+6% from switching to our Hopfield module" (125.05 to 132.85) mixes three things: learned β, this init difference, and the module itself. The new `bank0` run removes β from that comparison, and the `bank0+mhainit` runs (tasks 6 to 8) show how much of any remaining gap comes from the init.

## 2. Who runs what

Each workstream below runs on its own, with no waiting on anyone else. Work in your own copy of `src/` (for example `src_<name>/` or a git branch) so nobody overwrites anyone else's edits. When your part is done, put your result files, one table and one figure in the shared Drive folder `final-results/<your name>/`. Eshaan collects from there for the report, and Roshan for the deck. There are no other hand-offs.

If you can't write to Anuvik's CARC project folder, copy `env.sh`, set your own `ROOT`, and read data and checkpoints from his folder by absolute path. Reading is enough.

### Anuvik: generation (GPU) and retrieval analysis (CPU)

The CARC project folder is under Anuvik's account, so the new code goes in once, there. The lowdata job notes say GPU hours are nearly used up, so submit in this order.

```bash
# copy the four changed src files + tests/ + gen_controls.sbatch into $ROOT
cd $ROOT && $PY tests/test_bank_zero.py          # should print "6 checks passed"
sbatch gen_controls.sbatch                        # tasks 0-5: bank0 and bank64+beta, seeds 27-29
sbatch --array=6-8 gen_controls.sbatch            # bank0 with PyTorch-style starting weights
```

**bank64 ablations.** Copy `gen_controls.sbatch` and swap in three configs, `--bank_size 16`, `--bank_size 256` and `--bank_size 64 --steps 2`, with seeds 27 to 29 (9 runs). Add `bank16`, `bank256` and `bank64+steps2` to `VARIANT_PREFIXES` in `generation_test.py`.

**Convergence.** A run whose best epoch in `metrics.jsonl` is 19 or 20 out of 20 stopped at the cap. `train_generation.py` cannot resume, so rerun it with `--max_epochs 30`.

**Test set.** Check that the existing `bank0+beta` run folders (jobs 13233328 to 13233330) are inside `results/gen-20ng/`, named `gen-bank0+beta-seed2X-<job>`. If they live elsewhere, move or symlink them. Then:

```bash
$PY summarize.py results                          # validation table, attn is the reference row
$PY src/generation_test.py --results_base results/gen-20ng/ --data_dir data/20ng/ \
    --output results/generation-test-20ng.json
```

How to read the 2×2:

| | fixed β | learned β |
| --- | --- | --- |
| no bank | bank0 (new) | bank0+beta (132.85) |
| 64-slot bank | bank64 (144.89) | bank64+beta (new) |

Effect of the bank = bank64 − bank0 at fixed β, and bank64+beta − bank0+beta at learned β. Effect of β = the difference across each row's two cells. Module and init effect = bank0 − attn.

**Retrieval analysis (Banking77).** This uses the existing hmem checkpoints, so it can start on day one. Load each hmem `best.pt` the way `lowdata_test.py` does, including `refresh_memory` on the labeled subset. The hmem forward returns only the blended vector, so recompute the attention weights from the module's `q_proj`, `k_proj` and `log_beta`, averaged over heads. For each test query take the top memory slot and record two things: whether its label equals the true label, and whether it equals the model's prediction. Report both rates per size and seed, plus 3 worked examples (the query, its top 3 retrieved texts with labels, and the prediction).

### Eshaan: classification test set, extra seeds, statistics, retrieval controls (CPU)

**Extra seeds 30 and 31.** Start these first, since they take the longest. Copy `lowdata.sbatch` and change four lines so 2 datasets × 2 seeds × 2 sizes = 8 tasks map correctly:

```bash
#SBATCH --array=0-7
DATASETS=(banking77 20ng); SEEDS=(30 31); NS=(10 25)
DS=${DATASETS[$((i / 4))]}; SEED=${SEEDS[$(((i / 2) % 2))]}; N=${NS[$((i % 2))]}
```

**Classification test set.** Run `lowdata_test.py` once per dataset, and give each run its own `--output`, because the default path is shared and the second run would overwrite the first. Use seeds 27 to 31 once the extra seeds are in.

```bash
$PY src/lowdata_test.py --results_base results/lowdata-20ng/ --data_dir data/20ng/ \
    --seeds 27 28 29 30 31 --per_class_sizes 10 25 --output results/lowdata-test-20ng.json
$PY src/lowdata_test.py --results_base results/lowdata-banking77/ --data_dir data/banking77/ \
    --seeds 27 28 29 30 31 --per_class_sizes 10 25 --output results/lowdata-test-banking77.json
```

Gaps to fill:
- It computes accuracy only. Macro-F1 needs adding: collect predictions and use `sklearn.metrics.f1_score(..., average="macro")`.
- It already stores per-seed differences against the baseline. Add the 95% interval (paired t, or a bootstrap over test examples). This is the main results table.
- Pass `--per_class_sizes 10 25`. The default also asks for 100 and 200, which were never trained, and prints MISSING lines.

**Retrieval controls.** `src/lowdata_controls.py` takes one baseline run folder at a time, so it needs 12 calls: 2 datasets × 2 sizes × 3 seeds.

```bash
for DS in 20ng banking77; do for N in 10 25; do for S in 27 28 29; do
  B=$(ls -d results/lowdata-$DS/lowdata-baseline-n$N-seed$S-* | tail -1)
  $PY src/lowdata_controls.py --baseline_dir $B --seed $S --per_class $N \
      --data_dir data/$DS/ --results_base results/controls-$DS/
done; done; done
```

Gap to fill: it chooses k and β on validation and reports validation accuracy only. To compare with hmem's test numbers, score the test split with the chosen k and β, and save that next to `controls_val.json`.

### Roshan: efficiency benchmark (GPU)

Speed, memory and parameter count depend only on the architecture, not on the trained weights, so this runs on freshly built models without waiting for anyone's runs. Write `src/benchmark.py`. For each config record:
- the parameter count (`count_params`);
- test examples per second at batch 64, from one warm-up pass and the mean of 3 timed passes, with `torch.cuda.synchronize()` around the timer;
- peak GPU memory (`torch.cuda.reset_peak_memory_stats()`, then `torch.cuda.max_memory_allocated()`).

Configs: the classifiers baseline, hpool, hmem and hpool+hmem at each dataset's vocab size, and the decoders attn, bank0, bank0+beta, bank64 and bank64+beta. For hmem, fill the memory with random vectors of the labeled subset's size (10 or 25 per class), since the cost depends only on its size. Use one GPU type for everything and save one CSV.

### Madhulika and Saatwik: Track B and comparison models

The Track B code is not in this folder. Run the same subsets and seeds as Track A, so the analog and spiking numbers line up: `lowdata_subsets.make_nested_subsets`, seeds 27 to 29, 10 and 25 per class, on the `data/20ng/` and `data/banking77/` splits, with Track A's budget (up to 1,500 steps, evaluate every 50, stop after 10 evaluations with no gain). Measure your own models' speed and memory with the protocol in Roshan's section: batch 64, one warm-up pass, mean of 3 timed passes, same GPU type.

**Saatwik's TF-IDF reference.** scikit-learn `TfidfVectorizer` plus `LogisticRegression`, on the same nested subsets and seeds for both datasets, scored on the test split with accuracy and macro-F1.

**Saatwik's parameter-matched baseline.** On 20 Newsgroups, hpool+hmem has 467,976 more parameters than the baseline (8,785,436 vs 8,317,460). Each extra unit of feed-forward width adds 513 parameters per layer, 2,052 across 4 layers, so `ffn_dim` 1,252 instead of 1,024 brings the baseline to 8,785,316.
- In your own copy of `train_lowdata.py`, add an `--ffn_dim` option for the baseline and give the wide run its own variant name (for example `baseline_wide`).
- Check the count with `count_params`, and redo the arithmetic for Banking77, whose hmem label table is 77 × 256 instead of 20 × 256.
- Train 2 datasets × 2 sizes × 3 seeds on CPU with the `lowdata.sbatch` settings.
- Score the test split with your own short script (accuracy and macro-F1), since `lowdata_test.py` builds every encoder at width 1,024.

## 3. Method fact sheet (for the Methods section)

Numbers are from the midterm report and the Oct 20 deck. Check splits against each dataset's `config.json` before the final report.

| Item | Value |
| --- | --- |
| Datasets | AG News (Phase 1 pilot, 4 classes); 20 Newsgroups (20 classes); Banking77 (77 intents, one domain) |
| 20 Newsgroups split | 10,191 train / 1,123 val / 7,532 test; bydate split; headers, footers and quoted replies removed; val = stratified 10% of train, seed 544 |
| Banking77 split | 9,038 train / 965 val / 3,080 test; val = stratified 10% of the official train, seed 544; official test |
| Tokenizer | lowercase, regex `[a-z]+` or single punctuation mark (digits dropped); vocab capped at 20,000 words, built from training text only; pad 0, unknown 1; truncate or pad to 128 tokens. Banking77's training text has only about 2.3k distinct words, which is why its classifier is 3.79M parameters instead of 8.32M, and why the same added layers are a larger share there (up to 12.7%) |
| Low-label subsets | nested 10 ⊂ 25 per class, drawn per seed, so every variant at a seed sees the same examples |
| Backbone | 4 pre-norm layers, width 256, 4 heads (64 per head), GELU feed-forward 1024, learned positions, dropout 0.1 |
| Classifier | masked mean pooling + linear head; 8.32M parameters on 20 Newsgroups, 3.79M on Banking77 |
| hpool | mean pooling replaced by one learned query that retrieves over token states (Hopfield pooling); +264k parameters on 20 Newsgroups |
| hmem | retrieval over the labeled subset's own embeddings; values carry a label embedding; output added to the pooled vector before the head; +204k parameters on 20 Newsgroups |
| hpool+hmem | both; +468k parameters on 20 Newsgroups. Across datasets the variants add 2.5% to 12.7% |
| hmem memory rules | refreshed from the current encoder every 200 steps and at every evaluation; leave-one-out mask during training so an example never retrieves itself; at test time the memory is the labeled subset, with no mask |
| Decoder | same backbone, causal, untied 20k output head; 13.45M parameters |
| Decoder variants | attn (PyTorch attention); bank0 (Hopfield module, fixed β); bank0+beta (learned β per head, +16); bank64 (64 learned key/value slots per layer, fixed β, +131,072); bank64+beta (+131,088) |
| β | initialised at 1/√64 = 0.125 (same as attention scaling); fixed = buffer, learned = one log-β per head |
| Classification training | AdamW, lr 3e-4; up to 1,500 steps; evaluate every 50; stop after 10 evaluations without improvement; best checkpoint by val accuracy; CPU partition |
| Generation training | AdamW, lr 3e-4, weight decay 0.01, batch 64, gradient clip 1.0; up to 20 epochs; stop after 2 epochs without val improvement; GPU |
| Seeds | 27, 28, 29 for every configuration (30 and 31 planned for baseline and hmem) |
| Controls | frozen-encoder cosine kNN (k from {1, 3, 5, 10, 20} chosen on val); Hopfield readout softmax(β · cosine) over one-hot labels (β from {0.5, 1, 2, 5, 10, 20, 50} chosen on val) |
| Classification metrics | test accuracy and macro-F1, mean ± SD over seeds, per-seed differences against the baseline with 95% intervals |
| Generation metrics | test perplexity; perplexity without unknown-token targets; unknown-token rate in 64-step greedy decodes from 16-token prefixes of the first 50 val articles |
| Efficiency | parameter count, inference speed, peak GPU memory on one GPU type |

### Numbers so far (validation, midterm)

Classification accuracy, %, mean ± SD over seeds 27 to 29:

| Variant | 20NG, 10/class | 20NG, 25/class | Banking77, 10/class | Banking77, 25/class |
| --- | --- | --- | --- | --- |
| baseline | 19.7 ± 1.5 | 31.0 ± 1.0 | 67.9 ± 0.6 | 79.7 ± 0.8 |
| hpool | 17.4 ± 1.4 | 27.0 ± 1.4 | 63.0 ± 2.3 | 74.0 ± 0.4 |
| hmem | 19.5 ± 1.6 | 31.0 ± 1.3 | 67.4 ± 1.0 | 79.6 ± 1.8 |
| hpool+hmem | 19.4 ± 2.3 | 29.4 ± 0.6 | 66.1 ± 0.5 | 77.9 ± 0.6 |

Generation validation perplexity on 20 Newsgroups: attn 125.05 ± 1.44, bank0+beta 132.85 ± 0.44, bank64 144.89 ± 2.12.

## 4. Before you report anything

- Every headline number in the final report should be a test number, not validation.
- The earlier local "27% better" bank64 result was a false positive. Don't cite it.
- Keep run folder names as the scripts make them. `summarize.py` and the test scripts find runs by name.

## 5. What the unit test checks

1. With bank 0, fixed β and one step, the Hopfield layer gives the same output as `nn.MultiheadAttention` once the weights match (float64, with causal and padding masks).
2. `--force_hopfield` routes bank 0 through the Hopfield module, and without it bank 0 is PyTorch attention. Both have the same parameter count.
3. A whole bank0 decoder gives the same logits as the attn decoder when its weights are copied over.
4. Parameter differences match the report: bank0+beta adds 16, bank64 adds 131,072.
5. The default Hopfield init is wider than PyTorch's, and `init="mha"` matches PyTorch's.
6. Variant names come out as listed above.
