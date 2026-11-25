# Learning rules, plasticity, and recurrent computation

Research date: 24 November 2026. Three complete anchor readings; eight discovery-only references. This is an evidence review and collection of falsifiable hypotheses, not an implementation plan. No experiments, code changes, or commits were made.

## Assessment for BrainSwap

The best-supported distinction is between **how credit reaches a parameter**, **what changes during inference**, and **how much computation a prediction receives**. Predictive coding and equilibrium propagation address the first question. Test-Time Training (TTT) makes a sequence-specific model into recurrent state, addressing the second. Adaptive recurrence addresses the third. These mechanisms can interact, but evidence for one does not establish the others.

The 2017 anchors demonstrate principled alternatives to conventional backward computation in small supervised networks. Neither establishes better language modeling, lower total compute, or lower measured energy. Sun et al. provide direct language-model evidence for learned fast-state updates at 125M–1.3B parameters, especially as context grows, while retaining backpropagation in outer training. Their result motivates a controlled plastic-state experiment; it does not justify replacing BrainSwap's whole backbone.

The [midterm results](../docs/midterm-results.md) and [Phase 2 handoff](https://github.com/madhulikavikraman/BrainSwap/blob/1e5f89b/brainswap_phase2_handoff/PHASE2_HANDOFF.md) describe four-layer, width-256 models, 128-token examples, nested 10/25-label subsets, and seeds 27–29. Hopfield memory approximately ties classification baseline; pooling and the learned-bank decoder are worse in the reported validation runs. Initialization and temperature confounds remain central. Those are project findings, not results of this review. A learned static bank is not online plasticity. A 128-token experiment also cannot refute a benefit demonstrated primarily at thousands of tokens.

## Sources, versions, and reading coverage

All three anchor PDFs were accessible. `pdftotext`, `pdfinfo`, and installed Python PDF readers were absent at inspection; PyMuPDF was available through an isolated `uv tool run` environment. I extracted and read the entire PDF text in chunks, including references and every included appendix, and inspected selected rendered pages containing algorithms, equations, tables, and experimental plots. Extraction loses some mathematical glyphs and the predictive-coding algorithm image; the rendered page supplied that algorithm. The arXiv HTML provided a second representation for the two arXiv papers. No research-model code execution, dataset audit, cited-reference replication, or exhaustive search for separate supplements was performed.

| Anchor and exact primary source | Complete coverage | Source |
| --- | --- | --- |
| Whittington & Bogacz, *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity*, **Neural Computation 29:1229–1262 (2017)**, [DOI](https://doi.org/10.1162/NECO_a_00949), [Oxford-hosted published PDF](https://www.bndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf) | All 34 pages: §§1–4, including §§2.1–2.2.4, §§3.1–3.3, §§4.1–4.4; Algorithm 1, Figures 1–7, acknowledgments and references. **No appendix in this published PDF.** Not the 2015 bioRxiv precursor. | [Published article](https://www.bndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf) |
| Scellier & Bengio, *Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation*, [**arXiv:1602.05179v5**](https://arxiv.org/abs/1602.05179v5), submitted **28 March 2017**; PDF title page says 30 March | All 21 pages: §§1–6, every subsection; Appendix A, both gradient-lemma proofs and cost-reduction proof; B, constrained optimization; C, stochastic framework and proofs; references. [Versioned HTML](https://arxiv.org/html/1602.05179v5). | [v5 PDF](https://arxiv.org/pdf/1602.05179v5) |
| Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*, [**arXiv:2407.04620v4**](https://arxiv.org/abs/2407.04620v4), **31 August 2025** | All 33 pages: §§1–5, including §§2.1–2.7, §§3.1–3.3, §§4.1–4.3; contributions, references; Appendix A.1–A.4 dual derivation, B kernel estimator, C experiment details, Figures 13–16. [Versioned HTML](https://arxiv.org/html/2407.04620v4). | [v4 PDF](https://arxiv.org/pdf/2407.04620v4) |

The arXiv histories identify v5 and v4 as the displayed latest revisions when checked. TTT first appeared on 5 July 2024; v4 explicitly says all experiments were completed for the first version, with related-work and limitations updates thereafter. This review does not silently treat v4 as a new experimental replication. No anchor-body or included-appendix access gap remains; separately hosted code and additional materials have not been audited.

## Predictive coding: local signals can approximate backpropagation

### Biology and actual computation

Whittington & Bogacz motivate reciprocal cortical interactions, explicit mismatch populations, and synaptic changes based on local activity. Their actual algorithm is a layered Gaussian latent-variable model with **value nodes and separate error nodes**, iterative activity optimization, symmetric reciprocal weights, and Hebbian-form parameter updates. Locality describes information at a synapse after the network has communicated and settled. It does not imply that distant errors have disappeared or that communication is free. [Published article, §§2–3](https://www.bndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf).

In the paper's indexing, layer 0 is output and layer L is the clamped input. For prediction from the next higher layer, write

\[
\mu_l=\Theta_{l+1}f(x_{l+1}),\quad
\epsilon_l=\Sigma_l^{-1}(x_l-\mu_l),\quad
F=-\tfrac12\sum_l(x_l-\mu_l)^T\Sigma_l^{-1}(x_l-\mu_l).
\]

Here Σ contains fixed variances, not measured uncertainty estimates learned by this experiment. Maximizing F reduces variance-weighted mismatch. Inference and learning are, schematically,

\[
\dot x_l=-\epsilon_l+f'(x_l)\odot\Theta_l^T\epsilon_{l-1},
\qquad
\Delta\Theta_{l+1}=\alpha\,\epsilon_l^* f(x_{l+1}^*)^T.
\]

The first equation balances a node's own mismatch against errors arriving from below. The second correlates settled error with presynaptic output; it is a gradient-ascent update under the paper's negative-error convention. These summarize equations 2.11, 2.15, 2.17–2.19. The transpose and activation derivative are substantive requirements, not notation that eliminates weight transport.

With input alone clamped, an equilibrium reproduces the corresponding feedforward network and errors vanish. Training additionally clamps the desired output, altering hidden activities. At equilibrium, error recursion resembles backpropagation, but it is evaluated at altered activities. Exact correspondence therefore does **not** follow merely from writing both updates as error times activity. Increasing output variance relative to hidden-layer variances weakens target disturbance; rescaling the learning rate compensates for the shrinking signal. In that limit, updates approach backpropagation (§3.1). Perfect fit also gives matching zero updates, a much weaker observation.

### Evidence and limitations

The strongest evidence combines the analytical limit with an explicit gradient-direction comparison in a two-weight tanh task (300 synthetic samples), MNIST training, and a separate bidirectional noisy-association example. MNIST uses 784–600–600–10 networks, sigmoid activations, batch 20, Adam at 0.001, 20 inference iterations, and ten initializations. Output variances 1 and 100 are compared against backpropagation. All reach zero training error and roughly **1.7–1.8% validation error**; the authors did not optimize validation performance. Algorithm 1's simulation note initializes activities with a feedforward prediction before target clamping. Adam also adds practical optimizer machinery beyond the bare Hebbian rule. These are learning-trajectory comparisons, not equal-wall-clock efficiency experiments (§3.2, Figure 6).

The variance comparisons and gradient-angle maps are useful mechanistic controls. The two-branch linear example uses 2,000 training and 100 test samples, repeated 50 times: variance settings trade forward against inverse prediction, with ordinary regression strongest in its optimized direction (§3.3). Bidirectionality is demonstrated in this small model, not in text generation. There is no modern deep-language benchmark or systematic inference-step, feedback-asymmetry, precision, and energy ablation suite. The tiny gradient example can allow up to one million relaxation steps; that is an accuracy diagnostic, not an attractive training budget.

The authors explicitly acknowledge unsupported one-to-one value/error pairing, symmetric connectivity, negative error activities, nonlinear derivative mechanisms, and uncertainty over purely local convergence detection (§4.1). The backpropagation limit drives error magnitudes toward zero, creating a signal-to-noise problem for spikes (§4.2). Their anatomy discussion is a hypothesis, not causal evidence that cortex implements this rule. No hardware energy savings or LLM-scale efficiency follows from local plasticity.

### Bounded BrainSwap hypothesis

A small two-hidden-layer classifier over **identical frozen BrainSwap encoder features** can isolate credit assignment before attempting token-level training. One concrete success criterion is test accuracy within one percentage point of the backpropagation head, gradient cosine similarity above 0.9 on fixed diagnostic batches, and no greater measured training time, using at most 20 settling steps. These thresholds are proposed BrainSwap criteria, not paper results; macro-F1 must also be reported. Fixed features deliberately limit the conclusion to the head; they cannot show local end-to-end representation learning.

The strongest conventional comparator is the **same nonlinear head trained by backpropagation**, with separately selected learning rate; TF-IDF plus logistic regression remains a strong task-level reference for low-label text. Compare variance, iteration budget, and reciprocal-weight perturbations, recording gradient cosine similarity, convergence residuals, total matrix operations, latency, and paired test differences. Failure to match quality within the budget rejects the efficiency hypothesis even if eventual convergence occurs. An LLM conversion would require latent/error dynamics across causal sequence computations and a suitable categorical likelihood, not merely adding a reconstruction loss to the present decoder.

## Equilibrium propagation: gradients from two nearby equilibria

### Biology and actual computation

Scellier & Bengio propose using the **same neuronal dynamics** for prediction and error-driven settling. Their prototype is a continuous Hopfield-style energy model with symmetric weights, not BrainSwap's scaled-softmax Hopfield retrieval layer. Its central construction is

\[
F(\theta,x,s,\beta)=E(\theta,x,s)+\beta C(\theta,x,s),\qquad
\dot s=-\partial_sF.
\]

E specifies internal interactions, C the supervised cost, s hidden and output activities, and β the target's influence. First settle at β=0; then apply a small nonzero β and settle nearby. For free equilibrium s⁰ and nudged equilibrium sᵝ, the gradient theorem gives

\[
\nabla_\theta J=
\lim_{\beta\to0}\frac{
\partial_\theta F(\theta,x,s^\beta,\beta)-
\partial_\theta F(\theta,x,s^0,0)}{\beta},
\qquad J=C(\theta,x,s^0).
\]

For a pairwise Hopfield energy, a descent update becomes

\[
\Delta W_{ij}\propto
\frac{\rho(s_i^\beta)\rho(s_j^\beta)-
\rho(s_i^0)\rho(s_j^0)}{\beta}.
\]

The correlation difference estimates how a parameter affects cost through the equilibrium. It requires two states or integrated changes along their connecting trajectory. If C directly depends on θ, its direct derivative matters; the general F formula includes it. β is a **nudging coefficient**, not the retrieval inverse temperature also named β in BrainSwap. [Version 5, §§2–3](https://arxiv.org/html/1602.05179v5).

Appendix A derives the result by exchanging mixed derivatives of equilibrium energy and using stationarity; regularity and a nonsingular stable equilibrium branch matter. Appendix B recovers it through a Lagrangian, identifying the adjoint with sensitivity to β. Appendix C replaces equilibria with Boltzmann stationary distributions from Langevin dynamics; expected-cost derivatives require distributional expectations and the β→0 limit. It does not prove that a finite sample with incomplete mixing is an exact gradient estimator.

The STDP connection (§2.4) integrates a rate-change plasticity rule under tied reciprocal weights. It is a proposed bridge from timing-inspired plasticity to a gradient computation, rather than an experimental demonstration in spiking cortex. Phase control and symmetry remain biological liabilities.

### Experiments, controls, and mismatch with the theorem

The MNIST networks have one, two, or three hidden layers of 500 units. The dataset is described as 60,000 training and 10,000 test examples; Figure 3 calls the held-out curves validation error. Report the paper's **2–3% generalization-error range** without upgrading it to a separate untouched final-test protocol. Training error reaches zero (§5).

Table 2 exposes the cost: free-phase iterations are **20, 100, and 500** as depth increases; nudged iterations are 4, 6, and 8. State integration uses step 0.5, clipped hard-sigmoid activities, minibatches of 20, and Glorot initialization. Persistent per-example hidden states warm-start subsequent epochs, adding dataset-sized storage. No complete hardware-hours or measured-joules accounting is supplied.

Crucially, the experiments use **|β|=1**, randomly chosen signs, truncated nudged dynamics, and different learning rates per layer. These are meaningful practical accommodations. The theorem uses infinitesimal nudging and equilibria; the authors say a single learning rate should suffice theoretically and do not fully explain the layer-specific requirement (§5.2). Numerical approximation is their proposed explanation, not a resolved finding.

Depth comparisons and reported heuristics establish trainability, but there is no comprehensive factorial ablation of persistence, nudging magnitude/sign, phase truncation, and layer-specific rates, nor a modern matched-backpropagation efficiency result. Symmetric energy dynamics, long settling, branch changes, small-difference numerical cancellation, saturation, and noise can all complicate deployment. The discussion acknowledges that time-varying inputs remain an extension. Analog-hardware efficiency is a prospect; mathematical potential energy is not electrical consumption. [§§5–6 and Table 2](https://arxiv.org/html/1602.05179v5).

### Bounded BrainSwap hypothesis

An energy-based classifier on frozen Banking77 features would test whether nudging yields usable credit signals at low label counts. The strongest comparator for the learning claim is **backpropagation through the same finite unrolled dynamics**, with an implicit-equilibrium gradient reference where numerically feasible; an ordinary MLP head tests whether the energy model itself is worthwhile. Equal architecture alone does not equalize inference cost.

A concrete hypothesis is that at most 20 free and 8 nudged iterations yield gradient cosine above 0.9 and accuracy within one percentage point of the matched backpropagation reference. These are proposed acceptance thresholds, not established outcomes. Compare persistence on/off, β magnitude/sign, free/nudged step counts, and state resets. Count stored per-example activities and every relaxation. If persistence explains the apparent advantage, it is a memory/warm-start result rather than evidence of superior local credit assignment. Any future language-model version needs a causal state/target interface: a target-nudged state may train parameters, but cannot be reused to score the same unseen target as though it were a free prediction.

## TTT layers: learnable, sequence-specific plastic state

### Mechanism, equations, and LLM changes

Sun et al. use a small model's parameters as recurrent state. A fast model f is updated from current token features, while slow parameters learn which information its updates should preserve:

\[
\ell(W;x_t)=\|f(\theta_Kx_t;W)-\theta_Vx_t\|^2,\quad
W_t=W_{t-1}-\eta_t\nabla_W\ell(W_{t-1};x_t),\quad
z_t=f(\theta_Qx_t;W_t).
\]

K and V produce training and target views; Q produces the read query. W is sequence-specific state. The slow projections, initial W₀, token-dependent learning-rate gate, and backbone are learned through the outer next-token objective, including differentiation through inner gradients. The actual f includes normalization and a residual connection; TTT-MLP uses a two-layer GELU model with fourfold expansion. This is computational plasticity inspired by learning, not a validated model of synapses or a backpropagation-free learner. [v4, §§2.1–2.7](https://arxiv.org/html/2407.04620v4).

The proposed layer replaces a sequence mixer. It is not ordinary whole-model finetuning on a prompt. Mini-batches of 16 tokens compute gradients at a shared preceding state, but each output receives only its causal prefix of updates. The dual formulation uses matrix multiplications and a triangular mask instead of materializing every token's weight matrix. Appendix A derives equality with this mini-batch primal computation, **not equality with fully online gradient descent**. Appendix B's attention equivalence uses a nonparametric kernel estimator whose storage still grows with context. Reinterpreting attention as learning does not make ordinary softmax attention constant-memory.

### Experimental support and what it excludes

Experiments use 125M, 350M, 760M, and 1.3B models; Mamba counterparts are slightly larger. Pile contexts are 2k/8k; Books3 contexts range from 1k to 32k. Appendix C reports approximately 2.5B/7B/15B/26B training tokens, batch 0.5M tokens, AdamW, mixed precision, and a Chinchilla-style recipe. These are substantial training studies, far above BrainSwap's present budget. Baselines include a Llama-style Transformer, Mamba, and short-context Transformer training followed by long-context finetuning with 20% extra tokens and tuned learning rate/RoPE. FLOP-normalized comparisons should be read with these recipe differences visible.

At 2k, TTT-Linear, Transformer, and Mamba are broadly comparable; the MLP's added cost offsets its quality advantage in some comparisons. At longer contexts, TTT makes better use of context than the tested Mamba. Token-index perplexity in Figure 2 continues improving for TTT where Mamba plateaus after roughly 16k. This is stronger evidence than a neuroscience analogy, but it remains corpus-conditioned perplexity, not causal reasoning, reliable factual recall, or universal long-context superiority.

Table 1's sequential ablation moves from improved linear attention at 15.23 perplexity through normalization/residual design, mini-batch updating, learned update rate, and the Mamba backbone to 11.09 at 125M. Learnable initialization alone slightly hurts but stabilizes later configurations. Mini-batch updating supplies the largest individual decrement in that ordered chain. These are conditional ablations, not independent estimates that can be added across arbitrary backbones. Figure 7 measures the batch-size quality/runtime tradeoff. Backbone ablations reveal that temporal convolutions and gating contribute; credit cannot all go to plasticity.

Dual computation is over five times faster than their primal training implementation on TPUs. Reported TPU iteration times of 0.27 seconds for TTT-Linear versus 0.30 for Transformer at 2k are configuration-specific. A100-80GB inference comparisons use batch 16 for prefill and 512 for decode, with vLLM for Transformer. They do not establish interactive batch-one speed or energy savings. TTT-MLP suffers memory-I/O/runtime costs. Appendix C also documents checkpointing through time: fixed inference state does not imply constant training memory with naïve autodiff.

Figure 16 is an important counterweight: with fixed training recipes, overly long contexts worsen perplexity for models trained from scratch, and the preferred length depends on model size. No clean scaling-law fit supports extrapolation to arbitrary scale. v4 acknowledges prior delta-rule work and newer recurrent developments; the original experiments do not settle performance against all later baselines. No broad multiseed uncertainty analysis or measured energy study establishes the headline architecture comparisons as universal. [§3, Table 1, Appendix C](https://arxiv.org/html/2407.04620v4).

### Causality and a bounded test

Updating from xₜ before predicting **xₜ₊₁** is causal: xₜ is already observed. It would be leakage to reconstruct an unseen scoring target, update on it, and then score it. A vectorized chunk must not expose all chunk updates to early positions. Document reset boundaries, packed sequences, padding, and carried states are part of the task definition. A decisive diagnostic changes suffix tokens and requires all earlier logits to remain unchanged. Evaluation also needs independent per-stream states and no state carried from held-out-label tuning.

A small TTT-Linear-style state attached to BrainSwap's decoder would test whether learned writes beat **a strong delta-rule recurrent memory**, ordinary gated recurrence, and **causal dynamic evaluation** with equal state/update budgets. Keep a matched Transformer reference and compare frozen-state/write-disabled controls. Use the existing corpus first for short-context feasibility; a separate modest-length chronological or synthetic stream with delayed dependencies is needed for a memory claim. This is a hypothesis-generating adapter experiment, not a replication of the paper's fully trained architecture.

The falsifiable claim is improved held-out next-token loss or delayed factual recall at matched inference compute, without deterioration under contradictory/noisy updates. Measure reset-versus-carry behavior, interference, state bytes, update latency, and learning stability. A win only over a non-adaptive baseline does not show that meta-learned plasticity beats conventional adaptation.

## Adjacent angles: eight discovery-only primary references

These papers were identified through primary abstracts, metadata, or limited opening passages; **none was read in full for this report**. Their source links support followup, not counted as additional anchor readings. Statements below describe research leads and bounded questions, not endorsed effect sizes.

- **Active inference and uncertainty:** Friston et al., [*Active inference and epistemic value* (2015)](https://doi.org/10.1080/17588928.2015.1020053), [source](https://doi.org/10.1080/17588928.2015.1020053). Expected-free-energy policy selection combines preferences and information gain in simulations. A text agent could choose an informative query; compare ordinary value-of-information or uncertainty sampling at identical query cost. Prediction surprise can reflect irreducible noise, so it is not automatically useful novelty or calibrated epistemic uncertainty.
- **World models and embodiment:** Hafner et al., [*Mastering Diverse Domains through World Models*, arXiv:2301.04104v2, 17 April 2024](https://arxiv.org/abs/2301.04104v2), [PDF](https://arxiv.org/pdf/2301.04104v2). DreamerV3 learns action-conditioned models and imagined behavior in control environments. That supplies an interaction-based lead, not static-text LLM evidence. A bounded text-world comparison must equalize transitions, rewards, planning rollouts, and observation access; an ordinary model-based RL agent is the conventional control.
- **Neuromodulated writes:** Miconi et al., [*Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity*, arXiv:2002.10585v1, 24 February 2020](https://arxiv.org/abs/2002.10585v1), [PDF](https://arxiv.org/pdf/2002.10585v1). The abstract reports RL/supervised tasks and a parameter-controlled LSTM language-model comparison. Test a learned write gate against fixed, random, surprise-based, and ordinary recurrent gates with equal state. Neuromodulation is not evidence that a scalar dopamine analogue solves global credit assignment.
- **Reward prediction:** Schultz, Dayan & Montague, [*A Neural Substrate of Prediction and Reward* (1997)](https://doi.org/10.1126/science.275.5306.1593), [source](https://doi.org/10.1126/science.275.5306.1593). This primary theoretical synthesis connects primate dopamine observations to temporal-difference errors; it is not a modern intervention establishing an LLM mechanism. Delayed-reward text tasks could test eligibility-based updates against ordinary actor-critic learning. The publisher returned HTTP 403; the author-hosted paper was accessible and includes adjacent article residue on its first page.
- **Metaplasticity:** Benna & Fusi, [*Computational principles of synaptic memory consolidation* (2016)](https://www.nature.com/articles/nn.4401), [source](https://www.nature.com/articles/nn.4401). Coupled fast/slow synaptic variables motivate interference control. Test retention **and acquisition** against exponential decay, optimizer momentum, and replay at equal stored bytes. Their theoretical memory-capacity scaling does not establish LLM scaling or superior reasoning.
- **Developmental curriculum:** Bengio et al., [*Curriculum Learning*, ICML 2009](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf), [PDF](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf). Ordering examples supplies a separable training-data intervention. Compare easy-to-hard, reversed, and shuffled versions of the same examples with equal token exposure. Extra clean examples or environment feedback would confound an architectural claim; passive text ordering alone does not establish embodiment.
- **Adaptive recurrent reasoning:** Graves, [*Adaptive Computation Time for Recurrent Neural Networks*, arXiv:1603.08983v6, 21 February 2017](https://arxiv.org/abs/1603.08983v6), [PDF](https://arxiv.org/pdf/1603.08983v6). Learned halting allocates recurrent steps. Its abstract reports synthetic-task benefits but no large language-model gain. Compare learned halting with fixed recurrence at matched average **and maximum** block evaluations; measure latency and held-out compositional accuracy. More computation on difficult tokens alone is not proof of better reasoning.
- **Conventional online-adaptation baseline:** Krause et al., [*Dynamic Evaluation of Neural Sequence Models*, ICML 2018](https://proceedings.mlr.press/v80/krause18a.html), [source](https://proceedings.mlr.press/v80/krause18a.html). Gradient adaptation to recent sequence history is an essential alternative explanation for fast-learning benefits. Evaluate causally with identical resets, available tokens, update work, and tuning access. A TTT advantage must survive this comparison rather than rely solely on a frozen baseline.

## Interpretation and economical use of further review

Predictive coding is an inference/learning architecture for reducing latent prediction mismatch. Next-token prediction is a temporal statistical objective. Either can be paired with the other, but their shared word “prediction” establishes no algorithmic equivalence. Reward prediction error, reconstruction error, and Bayesian surprise likewise describe different quantities.

None of the three anchors establishes adaptive reasoning simply by running recurrent updates. A separate test needs controllable dependency depth, held-out compositions, and fixed-compute recurrence baselines. Similarly, lower FLOPs, local updates, spikes, and lower mathematical energy are distinct from measured joules. Causal attribution requires interventions on the proposed mechanism, not a model name or cortical analogy.

For token-conscious Gemini/Sonnet/Codex use, retain versioned sources and reuse this evidence ledger; ask a single available model to resolve a precise equation, appendix, or disputed baseline rather than commission repeated whole-paper summaries. No Gemini or Sonnet run was invoked here, and no model-specific price or reading-quality advantage is claimed. The present review required no subagents. Remaining uncertainty concerns empirical transfer and implementation cost, which further abstract summaries cannot settle.
