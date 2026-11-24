# Recurrent reasoning and additional architecture leads

Research date: 24 November 2026. This document distinguishes a complete text reading from targeted inspection and discovery. A complete reading does not imply replication, a formal proof audit, or independent verification of every cited work.

## Hierarchical Reasoning Model: complete text reading

Wang et al., [Hierarchical Reasoning Model, arXiv:2506.21734v3](https://arxiv.org/abs/2506.21734v3), 4 August 2025. [PDF](https://arxiv.org/pdf/2506.21734v3) and [HTML text](https://arxiv.org/html/2506.21734v3). Read all 24 PDF pages: sections 1–7, equations, pseudocode, figure captions, and references. All eight figures were also inspected in rendered PDF pages. This version contains no separate appendix. Training runs and source code were not reproduced; visual inspection does not verify the underlying data.

The mechanism couples fast and slow recurrent Transformer modules. Multiple low-level updates precede each high-level update; repeated segments refine the output. Training detaches earlier states and uses a one-step gradient approximation, deep supervision, and learned halting. These are concrete engineering choices motivated by cortical timescales, rather than evidence of implementing biological circuits.

The strongest evidence concerns grid reasoning. Its small-data results and full-data depth analyses use different training-set sizes. ARC evaluation includes evaluation demonstrations and inputs, augmentation, and voting over 1,000 transformed instances. This protocol must accompany any comparison with prompted LLMs. The reported constant activation-history memory does not imply constant runtime: recurrent computation still costs time. The biological dimensionality comparison is correlational, as the authors acknowledge.

For BrainSwap, test weight-shared recurrence against ordinary recurrent Transformers with identical supervision and compute. Separate recurrence, two timescales, gradient truncation, and halting. Evaluate language and reasoning independently. This paper does not establish improved general language modeling or show that biological hierarchy is necessary for its gains.

The model uses encoder-style blocks over problem grids. Adapting this to causal next-token generation requires an explicit masking and state-update design; directly reusing unrestricted input access could invalidate a language-model comparison. The fixed-point gradient rationale also depends on convergence assumptions, while practical training uses finite steps and a truncated approximation. Neither a universal-computation argument with unbounded resources nor a state-trajectory plot establishes useful, scalable language reasoning. In particular, claims about fixed-depth Transformer limitations need their precision and computation assumptions; they do not imply that autoregressive models with additional generated steps or external tools have the same restriction.

## Independent HRM critique: complete reading

Ge, Liao and Poggio, [Hierarchical Reasoning Model: A Critical Supplementary Material, arXiv:2510.00355v1](https://arxiv.org/abs/2510.00355v1), 30 September 2025. Read all nine pages, sections 1–7 and references, and inspected all four figures. There is no separate appendix. [PDF](https://arxiv.org/pdf/2510.00355v1) and [HTML text](https://arxiv.org/html/2510.00355v1). Experiments were not reproduced.

The authors report that enabling learned halting at evaluation improves Sudoku and maze results while reducing recurrent steps. Their eight-layer low-level-only variant also performs strongly, questioning whether two timescales explain the gains. This variant still retains recurrence, state, and the training procedure: it is not an ordinary one-pass Transformer baseline.

The evidence has limits: no multi-seed uncertainty, no ARC results in this version, and no full equal-compute attribution study. The Sudoku figures also show substantial checkpoint variation. The proposed diffusion/consistency interpretation is an analogy requiring further formalization, not a demonstrated equivalence of complete objectives and dynamics. The authors acknowledge conflicting outside findings about inference-time halting.

BrainSwap implication: test recurrence and halting separately from hierarchy. Count repeated training exposure as well as unique examples. Few distinct training puzzles does not mean little training compute. This critique strengthens the need for mechanism-specific controls; it does not establish that hierarchy never helps.

## Additional leads: targeted inspection, not complete reviews

These papers are worth investigating, but they do not count toward the full-reading corpus. Their claims remain author-reported. No ranking below is a measured performance comparison.

### Nested Learning / Hope

Behrouz et al., [Nested Learning, arXiv:2512.24695v1](https://arxiv.org/abs/2512.24695v1). Inspected the abstract, document structure, and the self-referential memory formulation around section 8.1; did not read all experiments or appendices A–C.

This work treats learning systems as interacting optimization processes at different update frequencies. It connects optimizers and memory, and proposes continuum memory and a self-referential sequence module. For BrainSwap, its value is the hypothesis that different memory components need different update schedules. A minimal controlled test would compare one versus two update rates at equal state size and update budget. Renaming an optimizer an associative memory does not itself demonstrate an empirical improvement. The full experimental and implementation claims need a dedicated reading before selecting Hope as a backbone.

### Dragon Hatchling / BDH

Kosowski et al., [The Dragon Hatchling, arXiv:2509.26507v1](https://arxiv.org/abs/2509.26507v1). Inspected the abstract, architecture outline, and portions of section 5 on sparse positive representations and modularity. Did not read all mathematical developments, experiments, or supplemental material.

The distinction to investigate is between local graph dynamics and their practical GPU formulation. Sparse positive activity, low-rank transformations, and attention-like sequence state provide specific hypotheses; the brain analogy is not an independent performance result. A useful control would isolate sparse positive activity from state updates and architecture width. Claims about scale-free structure, interpretability, and lifelong learning need separate operational tests. This review does not endorse BDH as an established improvement over modern LLMs.

### Dendritic attention

Yao et al., [Dendritic computation bridges neuroscience and foundation models](https://www.nature.com/articles/s41467-026-77929-8), Nature Communications 17, 11654, published 15 October 2026. Inspected the abstract and selected results/method descriptions around sparse write-in and block-diagonal memory; supplementary information was not reviewed.

The actionable ideas are input-dependent selective writes and structured memory interaction. The authors distinguish diagonal and dense state transitions and propose sparse or block-structured alternatives. For BrainSwap, compare learned selective writes with random writes, top-k gating, and a conventional gated linear-memory baseline at equal state size. Long context capacity must be separated from robust retrieval under interference and from downstream reasoning. The headline context length is not a guarantee that arbitrary information can be used reliably at that distance.

### NeuronSpark

Tang, [NeuronSpark, arXiv:2603.16148v2](https://arxiv.org/abs/2603.16148v2), revised 9 October 2026. Abstract and version metadata checked only.

The stated architecture combines input-dependent membrane dynamics, gated associative memory, and adaptive aggregation. A full review must separate their contributions and compare with strong selective state-space and linear-attention baselines. This is a particularly useful lead because it connects several BrainSwap interests. Hardware cycle/operation proxies, a physical FPGA demonstration, and practical GPU serving performance require separate evaluation; none substitutes for the others.

### Time-to-first-spike language models

Zhao et al., [Large Language Models with At Most One Spike per Neuron, arXiv:2609.05151v1](https://arxiv.org/abs/2609.05151v1), 4 September 2026. Abstract and version metadata checked only.

The authors explicitly report a language-modeling perplexity gap despite comparable performance on some understanding/reasoning evaluations. They describe energy as a spike-count proxy rather than measured neuromorphic consumption. This makes it a useful follow-up for temporal coding, but it does not establish a practical energy advantage for BrainSwap's hardware.

### Routed cortical columns

Khadangi, [Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns, arXiv:2602.22479v6](https://arxiv.org/abs/2602.22479v6), revised 17 April 2026. Abstract and submission history checked only.

The proposed system combines modular routing, selective retrieval, delayed writing, and replay consolidation. Its relevance is direct, but that combination makes attribution difficult. A full review needs per-component ablations, stream construction, causal update timing, matched active compute, and comparison with ordinary routed adapters plus replay. It is a research lead rather than independent confirmation that anatomical analogies improve models.

## Research questions these leads add

- **Computation versus storage:** Can extra recurrent steps recover an answer that is already represented, or does the model need additional memory capacity?
- **Temporal hierarchy:** Does separating update rates help beyond a single recurrent module given the same number of operations?
- **Write selectivity:** Does surprise predict information worth preserving, or mainly difficult/noisy tokens?
- **Memory interference:** What happens as facts accumulate, contradict one another, or share similar keys?
- **Mechanism attribution:** Does the biological component outperform the closest conventional implementation when width, state size, data, and compute match?
- **Systems feasibility:** Do sparse activations translate into lower measured latency and energy after indexing, routing, memory traffic, and control overhead?

These are hypotheses for future experiments. No training, benchmark runs, or architecture modifications were performed for this review.
