# BrainSwap: Testing Biological Neural Mechanisms for NLP

CSCI-544 (Applied NLP) course project, Group 27.

BrainSwap asks whether two brain-inspired mechanisms improve on a standard transformer when data, compute budget and evaluation are held the same:

- **Hopfield experiments (Track A).** Modern Hopfield associative memory ([Ramsauer et al., 2021](https://arxiv.org/abs/2008.02217)) added to a transformer:
  - Hopfield pooling (`hpool`)
  - a memory of labeled training examples (`hmem`)
  - a decoder with a learned memory bank (`bank64`)
- **Spiking experiments (Track B).** A spiking recurrent model vs an identical, parameter-matched analog (non-spiking) recurrent model.

We don't propose a new mechanism. We run a controlled, matched comparison that the literature usually skips, and we report negative results when that's what we find.

## Hypotheses

| | Hypothesis | Status at midterm |
|---|---|---|
| **H1** | Labeled-example memory helps classification most when data is scarce and classes are hard to tell apart (bigger gain on Banking77 than 20 Newsgroups, and at 10 than at 25 examples per class). | Not supported (validation) |
| **H2** | A decoder with 64 learned memory slots gets lower perplexity than standard attention. | Not supported; temperature and memory are confounded |
| **H3** | A spiking model stays competitive with its analog twin on classification but is worse on generation. | Supported on AG News only |

## Project phases

- **Phase 1 (done):** everything up to and including the midterm report. Pipelines, models, training and evaluation code, plus validation results on AG News, 20 Newsgroups and Banking77.
- **Phase 2 (in progress):** test-set evaluation, the missing controls, the generation 2x2 that separates memory from temperature, bank-size ablations, spiking experiments on 20 Newsgroups and Banking77, and efficiency benchmarks.

## Repository layout

```
BrainSwap/
├── CSCI-544 Project-27/            # Phase 1 (frozen; read-only)
│   ├── G27_CSCI 544 - Project Proposal.pdf
│   ├── BrainSwap_ Midterm Report (1).pdf
│   ├── BrainSwap_Report.docx
│   ├── BrainSwap_Midterm_Review.pptx
│   ├── src/                        # Phase 1 source code
│   └── brainswap-phase2-code/      # CARC runs of the expanded Phase 1 experiments
│       ├── src/                    # code exactly as run on CARC
│       ├── env.sh, setup_phase2.sh # CARC environment + one-time setup
│       ├── lowdata.sbatch          # classification job array (CPU)
│       ├── gen.sbatch              # generation job array (GPU)
│       ├── summarize.py            # validation summary tables
│       ├── results/                # run configs + metrics (no checkpoints)
│       └── logs/                   # Slurm logs
└── brainswap_phase2_handoff/       # Start of Phase 2 (bank-zero control)
    ├── PHASE2_HANDOFF.md           # commands, method fact sheet, rules: read this
    ├── src/                        # 4 updated generation files
    ├── tests/test_bank_zero.py     # 6 unit checks
    ├── gen_controls.sbatch         # missing generation 2x2 cells
    └── bank_zero.patch             # the real code changes, for review
```

> The folder named `brainswap-phase2-code/` holds the **expanded Phase 1** runs reported at the midterm, despite its name. Phase 2 work starts from `brainswap_phase2_handoff/`.

### Key source files

| File | Purpose |
|---|---|
| `src/data_pipeline.py`, `src/prep_real_data.py` | 20 Newsgroups download, cleaning, splits, tokenizer, vocab |
| `src/data_pipeline_banking77.py` | Banking77 download, splits, vocab |
| `src/lowdata_subsets.py` | Nested low-label subsets (10 ⊂ 25 per class), drawn per seed |
| `src/model.py` | Transformer encoder/decoder backbone and classifier |
| `src/hopfield_layers.py` | Modern Hopfield layer, `HopfieldPooling`, `HopfieldMemory` |
| `src/train_lowdata.py` | Low-data classification: `baseline`, `hpool`, `hmem`, `hpool+hmem` |
| `src/lowdata_test.py` | Test-set scoring of classification runs |
| `src/lowdata_controls.py` | Frozen-encoder kNN and Hopfield-readout retrieval controls |
| `src/generation_models.py` | Decoder variants: `attn`, `bank0`, `bank0+beta`, `bank64`, `bank64+beta` |
| `src/train_generation.py` | Language-model training with early stopping on validation perplexity |
| `src/generation_test.py` | Test perplexity, perplexity without unknown-token targets, greedy unknown-token rate |
| `src/train.py` | Full-data classification baseline (AG News pilot) |

## Setup

Requirements: Python 3.10+, PyTorch, NumPy, scikit-learn, `datasets` (Hugging Face).

```bash
pip install torch numpy scikit-learn datasets
```

### Data

Data isn't stored in the repo. Prepare it once (needs internet):

```bash
python src/prep_real_data.py --data_dir data/20ng/
python src/data_pipeline_banking77.py --data_dir data/banking77/
```

| Dataset | Classes | Train / Val / Test | Notes |
|---|---|---|---|
| 20 Newsgroups | 20 | 10,191 / 1,123 / 7,532 | bydate split; headers, footers and quotes removed |
| Banking77 | 77 | 9,038 / 965 / 3,080 | one domain, overlapping intents |
| AG News | 4 | — | Phase 1 pilot only |

Validation is a stratified 10% of train (seed 544). The tokenizer lowercases, keeps `[a-z]+` and single punctuation marks (digits are dropped), caps the vocab at 20k words from training text only, and truncates or pads to 128 tokens.

## Running

### Classification (Hopfield experiments)

```bash
python src/train_lowdata.py --seed 27 --per_class 10 \
    --data_dir data/banking77/ --results_base results/lowdata-banking77/ --variant all

python src/lowdata_test.py --results_base results/lowdata-banking77/ --data_dir data/banking77/ \
    --seeds 27 28 29 --per_class_sizes 10 25 --output results/lowdata-test-banking77.json
```

### Generation

```bash
# attn (PyTorch attention)
python src/train_generation.py --seed 27 --bank_size 0  --data_dir data/20ng/ --results_base results/gen-20ng/
# bank0: Hopfield module, no bank, fixed beta (needs the Phase 2 handoff code)
python src/train_generation.py --seed 27 --bank_size 0  --force_hopfield --data_dir data/20ng/ --results_base results/gen-20ng/
# bank64 with learned beta
python src/train_generation.py --seed 27 --bank_size 64 --learn_beta --data_dir data/20ng/ --results_base results/gen-20ng/

python src/generation_test.py --results_base results/gen-20ng/ --data_dir data/20ng/ \
    --output results/generation-test-20ng.json
```

### Unit tests (Phase 2 handoff)

```bash
cd brainswap_phase2_handoff && python tests/test_bank_zero.py   # expect "6 checks passed"
```

### On USC CARC

```bash
source env.sh                 # sets ROOT, PY, thread counts
bash setup_phase2.sh          # once, on the login node: env check, data, LOO patch
sbatch lowdata.sbatch         # classification, CPU partition
sbatch gen.sbatch             # generation, GPU partition
$PY summarize.py results      # validation tables
```

`env.sh` points to the group's CARC project paths. Edit `ROOT` and `PY` for your own account.

## Experimental protocol

| Setting | Value |
|---|---|
| Backbone | 4 pre-norm layers, width 256, 4 heads, GELU FFN 1024, dropout 0.1 |
| Seeds | 27, 28, 29 for every configuration |
| Low-label sizes | 10 and 25 examples per class (nested) |
| Classification budget | AdamW lr 3e-4; ≤1,500 steps; eval every 50; stop after 10 evals without improvement; best checkpoint by val accuracy |
| hmem memory | refreshed every 200 steps and at every eval; leave-one-out mask during training |
| Generation budget | AdamW lr 3e-4, wd 0.01, batch 64, clip 1.0; ≤20 epochs; patience 2 |
| β (inverse temperature) | initialised to 1/√64; fixed buffer or learned per head |
| Metrics | accuracy, macro-F1, paired per-seed differences with 95% intervals; test perplexity, non-unknown perplexity, greedy unknown-token rate; parameters, speed, peak memory |

The full method fact sheet is in [`brainswap_phase2_handoff/PHASE2_HANDOFF.md`](brainswap_phase2_handoff/PHASE2_HANDOFF.md) §3.

## Results so far (Phase 1, validation)

Classification accuracy (%), mean ± SD over seeds 27–29:

| Variant | 20NG 10/class | 20NG 25/class | Banking77 10/class | Banking77 25/class |
|---|---|---|---|---|
| baseline | 19.7 ± 1.5 | 31.0 ± 1.0 | 67.9 ± 0.6 | 79.7 ± 0.8 |
| hpool | 17.4 ± 1.4 | 27.0 ± 1.4 | 63.0 ± 2.3 | 74.0 ± 0.4 |
| hmem | 19.5 ± 1.6 | 31.0 ± 1.3 | 67.4 ± 1.0 | 79.6 ± 1.8 |
| hpool+hmem | 19.4 ± 2.3 | 29.4 ± 0.6 | 66.1 ± 0.5 | 77.9 ± 0.6 |

Generation validation perplexity on 20 Newsgroups: **attn 125.05 ± 1.44**, bank0+beta 132.85 ± 0.44, bank64 144.89 ± 2.12.

AG News pilot (spiking experiments): analog 88.99% vs spiking 89.05% accuracy; generation perplexity 152.36 vs 167.96.

**Summary:**
- Hopfield retrieval without external memory behaves like attention (identical predictions on all 12k AG News validation examples).
- Labeled-example memory ties the baseline.
- Hopfield pooling is worse and less stable.
- The memory-bank decoder is worse on generation.

These are **validation** numbers. Final reported numbers will be test-set results from Phase 2.

## Phase 2 roadmap

- [ ] Run `test_bank_zero.py` on the updated code
- [ ] Retrain the Phase 1 configurations (checkpoints are not in this repo) and score them on **test**
- [ ] Classification: macro-F1, paired 95% intervals, seeds 30–31 for baseline/hmem
- [ ] Controls: frozen kNN, Hopfield readout, TF-IDF + logistic regression, parameter-matched `baseline_wide`
- [ ] Retrieval analysis for hmem on Banking77 (top-1 label agreement, worked examples, memory-off check)
- [ ] Generation 2x2 (bank × β) + `bank0+mhainit`; ablations `bank16`, `bank256`, `bank64+steps2`
- [ ] Spiking experiments on 20NG and Banking77 (classification) and 20NG (generation), with spike rates and energy estimates
- [ ] Efficiency benchmark: parameters, throughput, peak GPU memory on one GPU type
- [ ] Final tables and figures

## Notes and known issues

- **Checkpoints (`best.pt`) aren't committed.** They live on CARC. `results/` holds only `run_config.json`, `metrics.jsonl` and `summary.json`.
- **Spiking-model code isn't in this repo yet.**
- An early local generation run that looked "27% better" was a **false positive**, overturned by the three-seed CARC rerun. Don't cite it.
- Three issues were found and fixed before the midterm results were accepted: the leave-one-out mask wasn't wired into hmem training (fixed and rerun), the false positive above, and a dataset mirror replaced with the original 20NG archive (checksum verified).
- Keep run-folder names exactly as the scripts produce them. `summarize.py` and the test scripts find runs by name.
- Our Hopfield module's default (xavier) init starts about 1.41× wider (q/k/v) and 1.73× wider (output) than PyTorch attention. `--hopfield_init mha` matches PyTorch.

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Ramsauer et al. (2021). *Hopfield Networks Is All You Need.* ICLR.
3. Weston et al. (2015). *Memory Networks.* ICLR.
4. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
5. Lv et al. (2023). *Spiking Convolutional Neural Networks for Text Classification.* ICLR.
6. Zhu et al. (2024). *SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks.* TMLR.
7. Roy, Jaiswal & Panda (2019). *Towards Spike-Based Machine Intelligence with Neuromorphic Computing.* Nature.
8. Casanueva et al. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* NLP4ConvAI @ ACL.
9. Lang (1995). *NewsWeeder: Learning to Filter Netnews.* ICML.
