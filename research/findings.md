# Brain-inspired mechanisms for improving language models

Research date: 24 November 2026. This is a critical literature review and set of experimental hypotheses, not a claim that a new model has been implemented or validated.

## Main conclusion

BrainSwap's most informative extension is to study **what gets written to memory, how quickly it changes, and how interference is controlled**. Its current experiments mainly study static learned slots and retrieval from a small labeled dataset. Those are useful controls, but they do not cover online plasticity, replay, or complementary fast/slow learning.

The practical recommendation is to preserve a strong ordinary language-model backbone while testing a small, separately controlled memory or adaptation mechanism. Replacing every component at once would make a positive or negative result difficult to explain. Spiking computation deserves a separate efficiency question tied to actual hardware. Predictive coding and local learning deserve a training-algorithm question tied to total inference and update cost.

This recommendation is our synthesis of the reviewed literature and repository, rather than an established result for BrainSwap. No experiments were run for this review.

The strongest objection is that ordinary retrieval, caching, and replay may already deliver most of the benefit. Fast parametric memory compresses observations and can lose exact detail; an explicit datastore can preserve it. Online updates also introduce interference and additional inference work. A memory extension should earn its place against those simpler alternatives, and a gain on newly introduced facts should be separated from general language-model improvement. The combination of familiar components is not a novelty claim.

## How to use the review

This folder contains the research findings. `findings.md` is the synthesis and reading ledger; the four topic reviews below contain the supporting analysis. [Learnings](learnings.md) captures the main scientific and review-method lessons. [Next steps](next-steps.md) turns the findings into the project's current plans, and [ideas](ideas.md) lists unscheduled directions. The optional [source cache](sources/cache-notes.md) keeps downloaded PDFs and extracted text out of Git. Public source links in the notes work without those local files.

- [Memory and continual learning](memory-review.md): associative retrieval, online memory, complementary learning systems, and replay.
- [Spiking and efficiency](spiking-review.md): temporal coding, surrogate-gradient learning, scale, and hardware claims.
- [Learning rules](learning-review.md): predictive coding, equilibrium propagation, and test-time training.
- [Recurrent reasoning and newer architecture leads](architecture-review.md): complete readings of HRM and an independent experimental critique, plus explicitly limited inspections of additional work.

Each specialist report records the papers and reading scope. A downloaded paper is not automatically a fully reviewed paper. A full reading means examination of the available paper text and identified appendices, not replication of results or verification of all references. Discovery-only and partially inspected papers must not be counted as full reviews.

The corpus is deliberately bounded. It combines foundational papers, direct language-model mechanisms, references already motivating BrainSwap, and a targeted search for newer work. Search was not a systematic database review with a preregistered query or complete citation graph. No finite review can establish coverage of every possible research angle. The map below exposes the main families and the questions that still need evidence.

## Paper-level reading ledger

Eleven distinct primary papers received body-and-included-appendix reviews. Figures and technical material were examined as documented in each report. This count excludes discovery-only references and the partially inspected 2026 work. It does not imply independent proof certification, exhaustive validation of bibliographies, reproduction of experiments, or auditing separately hosted supplements.

| Paper | Version | Review |
|---|---|---|
| [Hopfield Networks Is All You Need](https://arxiv.org/abs/2008.02217v3) | v3, 2021 | [Memory](memory-review.md); appendix proof assumptions and structure read, algebra not independently certified |
| [Titans](https://arxiv.org/abs/2501.00663v1) | v1, 31 December 2024 | [Memory](memory-review.md); a stated theorem has no proof in this version |
| [Complementary learning systems](https://doi.org/10.1037/0033-295X.102.3.419) | Published article, 1995 | [Memory](memory-review.md); body and simulations reviewed, bibliography not independently audited |
| [SpikeGPT](https://arxiv.org/abs/2302.13939v5) | v5, 2024 | [Spiking](spiking-review.md) |
| [Spiking CNNs for text classification](https://arxiv.org/abs/2406.19230v1) | arXiv v1, 2024; ICLR 2023 article | [Spiking](spiking-review.md) |
| [SpikeLLM](https://arxiv.org/abs/2407.04752v3) | v3, 2025 | [Spiking](spiking-review.md) |
| [Predictive coding with local Hebbian plasticity](https://doi.org/10.1162/NECO_a_00949) | Published article, 2017 | [Learning rules](learning-review.md) |
| [Equilibrium propagation](https://arxiv.org/abs/1602.05179v5) | v5, 2017 | [Learning rules](learning-review.md) |
| [TTT layers](https://arxiv.org/abs/2407.04620v4) | v4, 2025 | [Learning rules](learning-review.md); also a discovery reference in the memory report, counted once here |
| [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734v3) | v3, 2025 | [Architecture](architecture-review.md) |
| [HRM: A Critical Supplementary Material](https://arxiv.org/abs/2510.00355v1) | v1, 2025 | [Architecture](architecture-review.md) |

Sonnet handled associative memory and its completion pass; Gemini Flash handled spiking; Codex handled learning rules, architecture readings, cross-report checks, and synthesis, including checking spiking numerics and inference claims against primary sources. Model agreement was not treated as independent experimental confirmation.

## What BrainSwap currently establishes

The [midterm results](../docs/midterm-results.md) report validation experiments on small Transformers, low-label text classification, and short-context language modeling. Hopfield memory mostly ties the classification baseline, pooling is worse, and the learned-bank decoder is worse under the reported configurations. These observations constrain claims about those implementations and budgets; they do not establish that brain-inspired methods generally fail or that they improve large pretrained language models.

The Phase 2 handoff [Hopfield retrieval implementation](https://github.com/madhulikavikraman/BrainSwap/blob/1e5f89b/brainswap_phase2_handoff/src/hopfield_layers.py) makes the distinction concrete:

- A single retrieval step without a bank uses the same scaled-softmax attention form as ordinary multi-head attention. The handoff includes equivalence checks with matched weights.
- The learned key/value bank is updated by ordinary training and is fixed at inference. It is not a writable episodic store.
- Iterative retrieval changes the query in key space. It adds recurrent computation; it does not add a mechanism for storing new episodes.
- Classification memory stores embeddings of the labeled subset. It has different information access from an unsupervised stream memory or an external document collection.

The project's recorded confounds—initialization, inverse temperature, extra parameters, missing controls, and validation versus test reporting—remain essential. The word tokenizer and 128-token context also make this an imperfect test bed for long-context memory and modern pretrained-model adaptation. Better recall of information beyond a context boundary cannot be tested if the task never requires it.

## Coverage map

The following is a mechanism taxonomy, not a measured ranking. “Question” identifies what must be shown before claiming an LLM improvement. The specialist reports provide the relevant primary sources where reviewed; entries described as open do not imply that the entire literature lacks evidence.

| Research angle | Concrete model operation | Decisive question or control |
|---|---|---|
| Associative attractor memory | Similarity retrieval and optional iterative settling | Does it improve over matched attention or kNN with the same information? |
| Episodic external memory | Store and retrieve observations outside model weights | Does it beat ordinary retrieval at equal storage and retrieval cost? |
| Fast weights / online plasticity | Update a sequence-specific model state | Does adaptation help on unseen streams without future-token leakage? |
| Complementary learning systems | Fast episode capture plus slower integration | Does separation outperform one memory with the same total budget? |
| Replay / sleep-inspired consolidation | Revisit selected past observations | Does selection beat uniform or reservoir replay at equal replay tokens? |
| Synaptic consolidation | Constrain changes to important parameters | Does retention improve without preventing useful new learning? |
| Metaplasticity / multiple timescales | Change update rates or plasticity over time | Are gains due to update scheduling rather than more state? |
| Surprise / neuromodulated writes | Gate storage or learning using an error signal | Does the gate select useful novelty rather than noise and outliers? |
| Pattern separation / sparse coding | Make representations or writes less overlapping | Does reduced interference justify the added dimensions and routing? |
| Predictive coding | Iteratively adjust latent activities from prediction errors | Does total learning cost improve over backpropagation at equal quality? |
| Equilibrium / energy-based learning | Compare settled states or differentiate equilibrium behavior | Are settling, stability, and gradient approximations practical at LM scale? |
| Local credit assignment | Use local signals and eligibility traces | Can local updates learn long-range dependencies without hidden global machinery? |
| Spiking / temporal coding | Represent information in discrete events and timing | Does measured energy improve at matched quality and latency? |
| Dendritic computation | Structured state interaction or branch-selective integration | Does it beat a conventional gated or block-structured module? |
| Excitatory/inhibitory constraints | Restrict signs and balance activity | Do constraints improve a useful outcome beyond regularization controls? |
| Homeostasis / normalization | Stabilize activation and firing distributions | Does it add value beyond normalization and ordinary regularization? |
| Modular routing / cortical specialization | Activate selected subnetworks or adapters | Does it beat conventional sparse experts at equal active compute? |
| Structural plasticity | Add, prune, or reroute capacity during learning | Is improvement retained when parameter growth and search cost are counted? |
| Hierarchical recurrence | Allocate different update rates to interacting states | Does hierarchy beat ordinary recurrence with matched depth and supervision? |
| Oscillation / synchrony | Coordinate computation or communication across time | Is the timing mechanism causally useful rather than a descriptive analogy? |
| Cognitive maps / relational representations | Factor entities from reusable relations | Does transfer survive new entities, compositions, and held-out structures? |
| World models / active inference | Predict consequences and select information-gathering actions | Does the mechanism improve action or planning beyond a strong agent baseline? |
| Developmental curriculum / embodiment | Change experience ordering and interaction | Are gains due to architecture, curriculum, extra data, or environment feedback? |
| Brain-alignment objectives | Train representations toward neural measurements | Does increased neural similarity improve independent language tasks? |

Several families overlap. Surprise-gated sparse writes with replay and fast/slow states are a plausible combined system, but combining them before establishing individual effects would obscure the contribution. Neurotransmitters, oscillations, and cortical regions are biological descriptions; an experiment needs an explicit operation and a counterfactual intervention.

## Candidate experiments with falsifiable outcomes

These are independent research options, not an approved implementation sequence. Resource requirements depend on the chosen backbone, data, context length, and available hardware; no GPU-hour estimate has been invented.

Before investing in a new mechanism, resolve inexpensive diagnostics that affect interpretation of the existing results: matched initialization/temperature, memory-slot attention mass, key norms/separation, and the planned conventional retrieval controls. This preserves the value of BrainSwap's current experiments without assuming that a neutral result has a known cause.

### Writable episodic memory

Use a fixed pretrained backbone or a well-controlled small LM and an explicit key/value store of past observations. Compare no memory, conventional nearest-neighbor retrieval, and the proposed associative read at equal store size and candidate access. Include chronological facts, repeated entities, distractors, corrections, and paraphrased queries.

Measure answer quality, retrieval precision, useful information retained per byte, query latency, and failure under interference. Remove retrieved values or shuffle keys as diagnostic interventions. A gain that disappears against ordinary retrieval supports memory access as the benefit, not a novel associative mechanism.

### Fast/slow memory with consolidation

Separate a quickly writable episode buffer from a slowly updated adapter or memory module. Compare no replay, uniform replay, reservoir replay, and surprise-based replay at equal replay tokens and total update FLOPs. Evaluate on interleaved streams as well as obvious task boundaries.

Measure acquisition of new facts, retention of old capabilities, forward transfer, and adaptation/retrieval cost. Retention alone is insufficient: a system that refuses to learn can avoid forgetting. A biological selection rule earns its complexity only if it improves the acquisition–retention tradeoff over simpler selection.

### Small online adaptation module

Keep base weights fixed and compare a learned fast-state update with a simple delta-rule or linear-memory update. Add non-adaptive recurrence and a conventional lightweight adaptation control. Match state size; separately measure the cost of training the update rule and applying it at inference.

Respect chronology: predict the next token using only available inputs, score it, then perform any update that requires the newly observed token. Define whether memory resets per document, session, or stream. These choices describe different tasks. Include corrupted and contradictory observations to measure update stability.

### Adaptive recurrent reasoning

Compare one recurrent state with two interacting timescales, then isolate learned halting from fixed recurrence. Match repeated block evaluations, not merely parameter count. Distinguish language-model quality, exact task accuracy, and inference latency. Grid-puzzle success is a motivation for a language experiment, not its result.

### Spiking efficiency

Keep topology, training data, and evaluation matched to an analog recurrent or selective-state baseline. Include an appropriate low-precision conventional baseline. Report quality and the actual sequence processing protocol alongside spike rate and memory use.

Measure wall-plug or device energy when feasible, idle baseline, latency, throughput, precision, batch size, and hardware. Report operation-based energy estimates as estimates. A lower event count in a dense GPU simulation does not by itself demonstrate lower energy. Hardware-specific benefits can still be valuable even if the model does not win on a GPU.

## Evidence standards across all options

Use the same tokenization and data splits within a comparison. Match or report training tokens, trainable and total parameters, active compute, external storage, update work, and inference budget. A model given additional documents is answering a different information-access question unless baselines receive those documents too.

Select configurations on validation, freeze selection before final testing, report multiple seeds and paired differences, and retain per-run results. Account for repeated tuning and avoid treating a small-seed confidence interval as certainty. Report accuracy and perplexity alongside the capability being targeted: retention, interference tolerance, factual updating, long-distance recall, robustness, or energy.

Use counterfactual tests to establish mechanism: disable writes, freeze fast state, randomize replay selection, equalize compute, remove labels from retrieval values when appropriate, or replace the proposed module with its nearest conventional equivalent. An attractive neuroscience explanation should survive those substitutions.

## Additional conventional reference points

These are discovery references, not additional complete readings in this review:

- Khandelwal et al., [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172): an essential retrieval baseline.
- Kirkpatrick et al., [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114): a foundational consolidation/regularization reference; transfer to modern LLMs requires separate testing.
- [SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning](https://arxiv.org/abs/2511.22367): a direct follow-up lead for replay selection. Full methods and experimental review remains necessary before relying on its findings.

The architecture report lists additional late-2025 and 2026 primary papers with explicit access/reading limits. These discoveries keep the review from treating older foundational work as the current frontier, but they do not establish exhaustive coverage through the research date.
