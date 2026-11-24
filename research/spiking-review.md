# Spiking language models: mechanisms, evidence, and hardware limits

Research date: 24 November 2026. This report distinguishes results reported by the papers from our interpretations and proposed experiments. No models were trained or benchmarked for this review.

## Sources and reading scope

The specialist reviewed the full extracted text of three anchors, including their appendices. The parent agent checked bibliographic metadata, important tables, several implementation claims, and inspected the anchors' figures in rendered PDF pages. This is a literature reading, not a replication or certification of every equation. Generated text examples were inspected as examples, not treated as evidence of population-level generation quality.

| Anchor | Primary source and version | Coverage |
|---|---|---|
| Zhu et al., *SpikeGPT* | [arXiv:2302.13939v5](https://arxiv.org/abs/2302.13939v5), 11 July 2024; TMLR, June 2024 | 25 pages; main text, appendix A on LIF and backpropagation, B on datasets/baselines, C on RWKV, D/E visualizations and samples; figures 1–12 |
| Lv, Xu and Zheng, *Spiking Convolutional Neural Networks for Text Classification* | [arXiv:2406.19230v1](https://arxiv.org/abs/2406.19230v1), 27 June 2024; PDF identifies ICLR 2023 | 17 pages; main text, appendices A.1–A.6 on gradients, datasets, hyperparameters, theoretical energy, time steps, and limitations; figures 1–5 |
| Xing et al., *SpikeLLM* | [arXiv:2407.04752v3](https://arxiv.org/abs/2407.04752v3), 10 April 2025; ICLR 2025 | 20 pages; main text, appendices A.1–A.5 on quantization, calibration/training, ternary weights, GIF dynamics, and additional pipelines/results; figures 1–9 |

Downloaded PDFs are an optional, Git-ignored [local cache](sources/cache-notes.md). [SpikeLM, arXiv:2406.03287v1](https://arxiv.org/abs/2406.03287v1), ICML 2024, was also inspected by the specialist. It is an adjacent reference rather than an additional independently audited complete review. SpikeLM and SpikeLLM are different papers.

## Relation to BrainSwap

The [project README](../README.md) reports AG News pilot validation accuracy of 88.99% for the analog model and 89.05% for the spiking model, with generation perplexity 152.36 versus 167.96. Those observations do not identify their cause. The spiking implementation is absent from this checkout, so its exact neuron equations, coding scheme, masking, and training parity could not be audited here. No assumption that it duplicates SpikeGPT or the spiking CNN is justified.

The important distinction is between changing the recurrent backbone, changing the activation code, converting a pretrained network, and measuring a hardware implementation. Each anchor addresses a different combination.

## SpikeGPT: sequence-time spiking recurrence

### Mechanism and equations

SpikeGPT adapts RWKV-style recurrent token mixing and inserts leaky integrate-and-fire neurons. A membrane state integrates the current weighted input and decayed history, emits a binary event on crossing a threshold, and resets. Schematically, one common subtractive-reset LIF form is

`u_t = decay * u_(t-1) + W x_t - threshold * s_(t-1)`

`s_t = indicator(u_t > threshold)`.

This is an explanatory form, not a replacement for the paper's precise pre-reset/post-reset convention. Thresholding is nondifferentiable, so backward propagation uses a smooth surrogate derivative. Training still uses global backpropagation through time; using a spiking forward neuron does not establish local biological credit assignment.

RWKV retains recurrent sufficient statistics of decayed key/value contributions and gates their readout. The language sequence supplies the temporal axis: SpikeGPT avoids replicating each token across a separate large simulation-time dimension. Its state and some elementwise calculations remain continuous-valued. Binary emitted activity does not mean the entire model, its memory, or its output probabilities have only one bit.

### Evidence and compute

The 46M model was evaluated on Enwik8; the 216M model used OpenWebText2 pretraining. The paper reports four V100 GPUs and training durations of 12 and 48 hours for those models. These are training settings, not measured inference-energy comparisons.

Table 3 reports WikiText-103 test perplexity 39.75 for pretrained SpikeGPT-216M, versus 29.41 and 26.37 for the listed GPT-2 small and medium baselines. These differently pretrained models are contextual comparisons, not isolated tests of spiking. WikiText-2 uses an expanded training corpus including WikiText-103, which must accompany any comparison of those results.

The more informative matched architecture study is Table 5. Test BPC is 1.201 for vanilla RWKV, 1.403 for Heaviside RWKV, 1.306 for SpikeGPT-B, and 1.283 for SpikeGPT. It supports a quality cost for the tested binarized variants and a benefit from recurrent membrane dynamics relative to bare thresholding. The adjacent prose says SpikeGPT-B is better than SpikeGPT, which conflicts with the table; the numerical account here follows the table. This is an evidence inconsistency, not proof of an unavoidable binary-network performance floor.

Table 4 reports pretrained SpikeGPT-216M accuracy of 88.76% on SST-2, 51.27% on SST-5, 85.63% on MR, and 95.30% on Subj. These are classification results, not a broad reasoning benchmark. Figure 3 includes training curves at approximately 216M, 430M and 1.2B parameters over 0.9B tokens. A short token-budget scaling pilot cannot establish either a universal scaling ceiling or mature billion-parameter language quality.

### Efficiency and limits

Table 1 estimates energy using a measured firing-rate assumption and fixed arithmetic costs of 4.5 pJ per MAC and 0.9 pJ per accumulation. Its 32.2-fold ratio is a modeled quantity under those assumptions. It is not a measurement showing that a V100 or a deployed neuromorphic implementation consumes 32.2 times less energy.

The strongest contribution is demonstrating trainable spiking recurrent language generation with useful ablations. The remaining questions concern quality at adequate training budgets, state precision and storage, full-model operation coverage, and hardware execution.

**BrainSwap hypothesis:** compare an analog recurrent backbone with the same backbone under different threshold/reset choices. Vary surrogate gradient and firing rate separately. If the spiking variant remains worse after comparable tuning, that supports a cost of that coding/training configuration. It does not isolate quantization noise as the sole cause without additional controls.

## Spiking CNN: conversion and fine-tuning for classification

### Mechanism

The paper normalizes pretrained word embeddings into nonnegative firing probabilities, samples spike trains, converts a tailored TextCNN, and fine-tunes the converted SNN with surrogate-gradient backpropagation. The tailored network changes pooling and other architectural details to support conversion. The classification output combines activity from multiple neurons per class.

Unlike SpikeGPT, this model uses an additional simulation horizon: its default is 50 time steps per sentence. The relevant tradeoff is accuracy versus time steps, firing rate, and execution cost, rather than parameter count alone.

### Evidence and ablations

On MR, Table 1 reports original TextCNN 77.41 ± 0.22%, tailored TextCNN 76.94 ± 0.25%, and converted/fine-tuned SNN 75.45 ± 0.51%. On SST-2 the corresponding original and SNN values are 83.25 ± 0.16% and 80.91 ± 0.34%; on Subj, 94.00 ± 0.22% and 90.60 ± 0.32%. Conversion plus fine-tuning improves over the directly trained or merely converted SNN baselines in these experiments. Separating original from tailored TextCNN matters because conversion-compatible architectural changes already alter accuracy.

The appendices examine dropout, threshold/firing-rate sensitivity, fine-tuning duration, and inference steps. Increasing steps generally improves accuracy with diminishing returns; fine-tuning can obtain a given quality with fewer steps than conversion alone. These results support a tunable accuracy/work tradeoff, not a fixed universal time-step requirement.

The paper also reports stronger accuracy under several word-substitution attacks. This is evidence about the evaluated attack protocols. Establishing robust defenses requires adaptive attacks against stochastic prediction, matched attack budgets, and repeated randomness. Gradient masking is one possible concern, not an explanation established by these results. Several attacks operate through substitutions and model queries, so a gradient-only diagnostic would be insufficient.

### Energy accounting

Table 5 gives MR TextCNN 0.36 G FLOPs and SNN 5.49 G SOPs, with theoretical energies 4.498 mJ and 0.422 mJ. The 10.66-fold modeled advantage uses different assumed costs: 12.5 pJ per ANN operation versus 77 fJ per synaptic operation. The ratio 5.49/0.36 is about 15.25, but FLOPs and SOPs are not interchangeable units of hardware work.

This explains how more counted synaptic events can coexist with lower estimated energy. It does not by itself invalidate an event-driven hardware advantage, and it does not measure a GPU advantage. Storage, routing, conversion, neuron-state updates, and utilization need consideration in an end-to-end measurement.

**BrainSwap hypothesis:** measure matched analog/spiking classification across both coarse and fine-grained datasets, at several time-step and firing-rate settings. Define a noninferiority margin before testing. Class count alone does not imply that spikes lack enough semantic capacity; a binary population can encode many distinct states.

## SpikeLLM: saliency-aware spiking as low-precision representation

### Mechanism

SpikeLLM adapts pretrained LLaMA models through quantization/calibration pipelines rather than training a large temporal SNN from random initialization. Generalized integrate-and-fire (GIF) neurons merge steps into multilevel outputs. Optimal Brain Spiking allocates additional representational steps to salient channels using sensitivity information. A ternary variant supports additive linear-layer operations.

This connects spiking to mixed precision and quantization, but does not make it identical to uniform integer quantization or BitNet. Residual state, channel selection, step allocation, scales, and training/calibration procedures can differ. Two GIF steps cannot be assigned a universal integer bit width without counting their exact level set, state, scale, and encoding.

### Evidence, controls, and limits

The paper evaluates pretrained models from 7B to 70B using OmniQuant and GPTQ-related pipelines. The OmniQuant calibration examples are 128 WikiText-2 chunks of 2,048 tokens. Appendix A.2 includes 16 paired resamplings of calibration data for one comparison; most other settings use a fixed seed. This is stronger than a single calibration sample, but not universal seed-level replication of every result.

Table 3 reports a LLaMA-7B W4A4 gain over the paper's OmniQuant setup; Table 4 combines the approach with QuaRot. Figure 4 includes approximately equal-operation comparisons and random-saliency controls. Those controls must be acknowledged before asserting that the benefit is merely extra precision.

For additive linear layers, Table 6 reports average zero-shot accuracy 52.10% for one SpikeLLM ternary configuration and 38.98% for the listed PB-LLM90% configuration. Different step allocations change quality and work, so the headline difference is not a general ranking of ternary methods. Table 5 reports ACE and sparse-ACE operation proxies, not measured joules. The paper also explores QLoRA and efficient-QAT settings in the appendix.

The 70B caveat is specific: W4A4 and W2A8 results are not evaluated for LLaMA-2-70B in the OmniQuant pipeline because of stated GQA-related training instability. This is not failure of every 70B spiking configuration.

**BrainSwap hypothesis:** compare saliency-based GIF with saliency-based conventional mixed precision, including matched representation/storage, approximate operation cost, calibration access, and kernel implementation. If conventional precision allocation matches quality and measured cost, a distinct benefit from spike dynamics remains unestablished. If it does not, inspect residual accumulation and the step-dependent quantizer rather than attribute the gain automatically to biological realism.

## Cross-paper conclusions

- **There is no established one-billion-parameter impossibility threshold.** SpikeGPT itself includes a larger-scale training pilot. The separately discovered [NeuronSpark v2](https://arxiv.org/abs/2603.16148v2) also reports billion-scale from-scratch pretraining, although that paper has not received a full review here.
- **Surrogate gradients are approximate learning rules.** They can introduce optimization difficulties; the anchors do not prove inevitable failure beyond a particular sequence length or model size.
- **Simple leaky integration is related to recurrent state-space updates.** A scalar LIF neuron is not identical to all selective SSMs. Reset and gating can complicate parallel algorithms, but recurrence alone does not rule out parallel scans, chunking, or specialized kernels.
- **Hardware outcomes are empirical.** GPUs may handle irregular sparse updates poorly, yet dense or fused implementations need not branch on every spike. Neither universal GPU energy superiority nor universal inferiority follows from event counts.
- **Perplexity is not rollout error.** Under ordinary teacher-forced evaluation, errors are scored against observed prefixes. Compounding mistakes in freely generated continuations are a different issue. The AG News perplexity gap does not establish either mechanism.

## A practical measurement and evaluation protocol

Compare the existing matched recurrent analog and spiking models, then add a strong conventional low-precision baseline where appropriate. Report quality, active/total parameters, precision, state size, spike rate, batch size, sequence length, prefill and decode latency, throughput, and device model.

Measure gross device energy and, if useful, idle-subtracted incremental energy. NVML samples device telemetry, not wall-plug power; respect the sensor's actual update rate, use sufficiently long repeated trials, and report uncertainty. Include CPU/system energy or wall-plug measurements if claiming system energy. An operation proxy remains useful for architectural analysis when clearly labeled and paired with its assumptions.

For mechanistic ablations, hold topology and data constant while varying thresholding, reset, surrogate gradient, state precision, and internal steps. For classification parity, use a predefined practical equivalence or noninferiority margin; a nonsignificant difference is not proof of equivalence. For generation, report both likelihood-based metrics and an independently specified downstream capability.

## Broader angles: discovery rather than complete reviews

The following are research connections, not established evidence that the corresponding biological mechanism improves LLMs:

- **Sparse experts:** [Switch Transformers](https://arxiv.org/abs/2101.03961) provides a conventional conditional-computation comparator. Its success does not validate a cortical mechanism.
- **Dendritic integration:** input-dependent branch selection and structured state updates are concrete hypotheses. The [2026 dendritic-attention paper](https://www.nature.com/articles/s41467-026-77929-8) is a targeted lead documented in the architecture report; it is not a complete reading here. A gated MLP is not automatically a biological dendrite.
- **Excitation/inhibition:** fixed-sign connectivity and activity balance could be tested as constraints. The reviewed anchors do not establish general LLM gains or inevitable harm from those constraints.
- **Homeostasis:** compare local activity/firing-rate regulation against ordinary normalization and regularization. Transformer normalization is not evidence that a particular biological homeostatic rule works.
- **Structural plasticity:** distinguish fixed sparse inference from learning connectivity online. Count regrowth/pruning overhead and changing capacity.
- **Oscillations:** test an explicit timing or communication mechanism. RoPE is a positional transform, not evidence of communication through biological synchrony.
- **Ternary representations:** [BitNet b1.58](https://arxiv.org/abs/2402.17764) is a relevant conventional reference. Shared numerical symbols do not establish equivalent architectures, optimization, or hardware costs.
- **Neuromorphic systems:** [Intel's Loihi 2 technology brief](https://www.intel.com/content/www/us/en/research/neuromorphic-computing-loihi-2-technology-brief.html) is a hardware follow-up lead. Capacity, precision, routing, software support, and system scale must be assessed for a specific mapping; neuron counts cannot be equated directly with LLM parameter counts.
