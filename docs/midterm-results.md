# Midterm results (validation only)

These are the midterm's **validation** numbers. They motivate the current direction. They are not test results, and the final presentation must not report them as test results. The code, run configs and Slurm logs that produced them are at commit [`1e5f89b`](https://github.com/madhulikavikraman/BrainSwap/tree/1e5f89b), in `CSCI-544 Project-27/brainswap-phase2-code/`.

## Setup

- **Models.** Small Transformers trained from scratch: 4 pre-norm layers, width 256, 4 heads, a word-level tokenizer with a 20k vocab, and 128-token inputs.
- **Classification.** Nested low-label subsets of 20 Newsgroups and Banking77, with 10 and 25 examples per class.
- **Generation.** 20 Newsgroups.
- **Seeds.** 27, 28 and 29.

| Variant | Mechanism |
|---|---|
| `hpool` | Hopfield pooling: one learned query retrieves over token states |
| `hmem` | Hopfield retrieval over the labeled subset's own embeddings |
| `bank0+beta` | Hopfield attention module with learned inverse temperature β, no memory slots |
| `bank64` | 64 learned key/value slots per layer, fixed β |

## Classification accuracy (%), mean ± SD over seeds 27–29

| Variant | 20NG 10/class | 20NG 25/class | Banking77 10/class | Banking77 25/class |
|---|---|---|---|---|
| baseline | 19.7 ± 1.5 | 31.0 ± 1.0 | 67.9 ± 0.6 | 79.7 ± 0.8 |
| hpool | 17.4 ± 1.4 | 27.0 ± 1.4 | 63.0 ± 2.3 | 74.0 ± 0.4 |
| hmem | 19.5 ± 1.6 | 31.0 ± 1.3 | 67.4 ± 1.0 | 79.6 ± 1.8 |
| hpool+hmem | 19.4 ± 2.3 | 29.4 ± 0.6 | 66.1 ± 0.5 | 77.9 ± 0.6 |

## Generation validation perplexity, 20 Newsgroups

attn 125.05 ± 1.44, bank0+beta 132.85 ± 0.44, bank64 144.89 ± 2.12.

The bank0+beta vs attn gap mixes three things: learned β, a wider default initialization, and the module itself. The midterm runs do not separate them.

## AG News pilot (spiking vs analog recurrent model)

Accuracy: analog 88.99%, spiking 89.05%. Generation perplexity: analog 152.36, spiking 167.96.

## What these results establish

- One-step Hopfield retrieval without external memory behaves like attention. It gave identical predictions on all 12k AG News validation examples, as Ramsauer et al. predict.
- Labeled-example memory (`hmem`) ties the baseline. Hopfield pooling is worse and less stable. The learned-bank decoder is worse on generation.
- Every memory tested is **read-only**. The bank is fixed after training, and the experimenter fills `hmem`. None of these runs tests what to write, when to write, or how to handle interference. [Next steps](../research/next-steps.md) targets that gap.

An early local generation run that looked "27% better" was a false positive, overturned by the three-seed rerun. Do not cite it.
