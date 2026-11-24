# Memory mechanisms for LLMs: Hopfield, Titans, complementary learning systems

Scope: three anchor papers reviewed across their bodies, included appendices, and figures (bibliography and verification limits in §0); eight adjacent papers used for discovery only; and what each anchor says about BrainSwap's memory variants. This file contains no implementation plan.

## 0. Sources and coverage

| Role | Paper | Version / identifier | Primary source |
|---|---|---|---|
| Anchor 1 | Ramsauer et al., *Hopfield Networks is All You Need* | arXiv 2008.02217 **v3**, 28 Apr 2021 (first posted 16 Jul 2020) | [Paper](https://arxiv.org/abs/2008.02217v3) |
| Anchor 2 | Behrouz, Zhong, Mirrokni, *Titans: Learning to Memorize at Test Time* | arXiv 2501.00663 **v1**, 31 Dec 2024 (no later version on the abs page when fetched) | [Paper](https://arxiv.org/abs/2501.00663v1) |
| Anchor 3 | McClelland, McNaughton, O'Reilly, *Why there are complementary learning systems…*, Psychological Review 102(3), 419–457 (1995); DOI 10.1037/0033-295X.102.3.419 (checked against Crossref) | Author-hosted published PDF (39 pp.) | [PDF](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf) |

No substitution was needed for anchor 3: the author PDF is accessible.

**Extraction and inspection.** Text came from PyMuPDF. Every figure page of all three papers was rendered and inspected (images kept outside the repo). Values read off plots are approximate and marked "≈". The CLS PDF is a page scan with an OCR text layer; its equations (3)–(9) and Table 1 were checked against the page image.

**Coverage.**
- *Hopfield (94 pp.).* Main body, appendix A.1–A.6 including the convergence, fixed-point and capacity proofs (A.1.4–A.1.6), the A.2 lemmas and their included proofs, the binary-network derivation (A.3.2), the PyTorch description and flowchart (A.6.1–A.6.3), and Figures 1–5 and A.1–A.7. For proofs, statements, assumptions and proof structure were read; intermediate inequalities were not independently re-derived. A.2 uses softmax Jacobian bounds and positive semidefiniteness, convex conjugacy, the Lambert-W definition/identities, and an integrated-Jacobian mean-value theorem. Bibliographic entries were inspected but not comprehensively validated or traced to their sources.
- *Titans (27 pp.).* Body (§1–§6), Appendices A–C, Tables 1–5 and Figures 1–9. Theorem 4.1 has no proof in this version. The paper gives no code and no hardware description. References were not checked.
- *CLS (39 pp.).* Pages 1–35 (clinical lesion review, the account of the two systems, evidence, interleaved learning, catastrophic interference, slow-learning argument, all four consolidation fits, the two-store model, age/species discussion, general discussion, conclusion) and Figures 1–14. Pages 36–39 are references and were not read.
- Across all three: cited raw data and algebra were not independently certified. A complete technical reading does not mean that every cited source was retrieved or that every theorem was independently proved.

## 1. What BrainSwap's memory variants actually are

Checked against `hopfield_layers.py` and the Phase 2 fact sheet.

| Variant | Where information lives | Who writes it | When | Read operation | Forgetting / consolidation |
|---|---|---|---|---|---|
| `bank64` | `k_bank`, `v_bank` parameters of shape (heads, slots, d_head), concatenated to the layer's keys and values | SGD on the training loss | training only; frozen at inference | softmax read, always visible to every query | none |
| `hmem` | embeddings of the labeled subset plus label embeddings | periodic re-encoding of the labeled training subset by the current encoder (every 200 steps and at each eval) | training; at test the memory is the same labeled subset | softmax read with leave-one-out mask in training | none |
| `hpool` | learned static queries | SGD | training only | softmax pooling over the sequence | none |

Consequences:

1. `bank64` is **learned static slots**: a persistent, task-level, semantic store, written by the slow learner. It is not an episodic memory. The model never writes to it in response to what it reads at test time.
2. `hmem` is a **read-only external store populated by the experimenter**, equivalent in kind to a kNN or prototype head over the model's own embeddings. The model never decides what to write.
3. Neither variant tests the CLS claim (fast store plus slow store plus replay), the Titans claim (test-time gradient writes with decay), or the Hopfield claim about capacity (patterns stored in a content-addressable memory). `bank0 ≡ attn` is exactly what Hopfield Eq. 10 predicts, so that result is a consistency check rather than a finding about memory.
4. The H1 result (hmem ties baseline) has several candidate explanations that these papers cannot separate. hmem stores the same labeled examples the classifier trains on, so it may add little beyond the weights; but training on an example does not mean the weights have absorbed it, so this is a hypothesis, not an explanation. Others: encoder quality and re-encoding lag, softmax sharpness and key separation relative to the number of stored items (Hopfield: retrieval needs β·Δ well above ln N, §2), the leave-one-out mask, and the low-data regime. CLS predicts the fast store pays off for items **not yet absorbed into the slow weights** and under **sequential arrival**; BrainSwap's setup tests neither.

## 2. Anchor 1: Ramsauer et al., modern Hopfield networks (arXiv 2008.02217v3)

**Biological mechanism vs algorithm.** The biological referent is Hopfield's 1982 attractor network (content-addressable recall by energy descent). The paper makes no claim about hippocampus, replay or any specific brain region; "brain" appears only in passing. The contribution is mathematical: a continuous-state Hopfield network whose one-step update coincides with transformer attention. Brain-inspiration is nominal.

**Equations, in words.**
- Energy (Eq. 2): negative log-sum-exp of the query's similarity to N stored patterns, scaled by inverse temperature β, plus half the squared norm of the state, plus constants; for states inside the simplex of the patterns the energy lies between 0 and 2M² (M = largest pattern norm). The quadratic term keeps the state bounded.
- Update (Eq. 3): the new state is the stored-pattern matrix times a softmax over β-scaled similarities. Global convergence (A1/A2) is a concave–convex-procedure argument (Zangwill/Sriperumbudur–Lanckriet): the energy decreases strictly, step lengths go to zero and limit points are stationary points, which may be saddles. It does not promise convergence to a stored pattern; the iterate converges if the stationary set is finite.
- Eq. 10: with queries/keys projected by `W_Q`, `W_K`, values by `W_V`, and β = 1/√d_k, the update equals scaled dot-product attention. Retrieval is a convex combination of stored values. A transformer applies **one** step, so the fixed-point results matter only when one step already lands near the fixed point.
- Fixed-point taxonomy and its assumptions (A.1.5). Single-pattern fixed point in a ball of radius 1/(βNM) around xᵢ: Δᵢ ≥ 2/(βN) + β⁻¹·ln(2(N−1)NβM²). Metastable state of l similar patterns: spread m_max with β·m_max² ≤ 1 and the cluster centre separated from the other patterns by a comparable margin; the fixed point sits within about m_max of the centre. Global fixed point near the mean: β·m_max² < 1 (contraction), and if the data are centred the mean is itself a fixed point (the paper notes that normalisation layers promote this). One-step retrieval error ≤ 2(N−1)M·exp(−β(Δᵢ − 2‖ξ−xᵢ‖M)): retrieval needs βΔᵢ comfortably above ln N, which is a condition on key separation and sharpness, not on capacity alone.
- Capacity (Thm 3 and A3–A7): "stored" means a fixed point inside the sphere of radius 1/(βNM) around each pattern, with disjoint spheres. The guaranteed basin therefore shrinks as N grows: exponential capacity is the capacity of well-separated codes, not robust completion from noisy cues. The paper states this trade-off (A.1.6) and the binary Demircigil result has the same property (attraction radius vanishes at maximal capacity). For random patterns on a sphere of radius K√(d−1) with β = 1, N ≥ √p · c^((d−1)/4); the paper's constants are c ≥ 1.37 (d = 75, K = 1, p = 0.001) and c ≥ 3.15 (d = 20, K = 3). The √p factor makes the guaranteed N at the d = 75 constants about ten patterns (arithmetic from the stated constants). Pattern norms in the theorem grow with √d, while attention uses β = 1/√d_k with learned key norms, so the theorem does not transfer to learned projections at d_head = 64.
- Forgetting (A.1.8.2): a recency penalty γ·(age) added inside the log-sum-exp of the energy. This is a sketch: as printed, the update rule (Eq. 453) omits the γ term that the energy (Eq. 452) contains, and no experiment uses it.

**What changes in an LLM.** Almost nothing at the attention level, since attention already is the update. The new content is three layer types (`Hopfield`, `HopfieldPooling`, `HopfieldLayer`) that choose where queries and stored patterns come from: input sets, learned static queries, or learned/fixed stored sets. Options include multi-step iteration, learned β, and pattern normalization (default: layer norm on the inputs, none after projection), which the paper identifies as the control on pattern norm M and centring and hence on metastable-state size. `HopfieldLayer` may hold the training set, a reference or prototype set, or a learned set as its stored patterns, which places `hmem`-style and `bank`-style memories inside the family the paper describes (it compares the layer to kNN/SVM behaviour).

**Experiments and compute.**
- *BERT head analysis (analysis only, no new model quality claim).* Heads are classed by how many softmax entries sum to 0.9 (k). In pretrained BERT-base, lower layers average over many tokens, middle layers use few and the last layers a medium number. In BERT-small after 1.45M steps (128-token sequences, bidirectional encoder), the Fig. A.4 violin plots show median k of roughly 1–32 in layers 4–12 and roughly 13–93 in layers 1–3 (≈, read from the plot). Heads that commit to small metastable states do so early (~9–10k steps) and then see small softmax Jacobians. So trained heads already act as sharp retrievers in most layers and as averagers in the lowest, which is the baseline any extra memory slots must compete with. BERT-small: 12 layers, 4 heads, width 256; about four days on one V100 for 1.45M steps. "Gaussian averaging heads" (input-independent attention with learned location and scale) appear only in appendix A.5.1; the quoted ~95× parameter reduction is for BERT-base (d = 768, d_k = 64, N = 512), and no quality result is reported.
- *Immune repertoire classification (DeepRC).* ~300k instances per repertoire. Real CMV data AUC 0.832 ± 0.022 vs SVM with MinMax kernel 0.825 ± 0.022: within one SD. The large gaps are on the 0.1%-signal implanted settings (0.98 and 0.94 vs 0.58 and 0.53 for the MinMax SVM; 5 CV folds). Baseline numbers come from the DeepRC papers, not re-run here.
- *MIL benchmarks.* AUC best on Tiger (91.3), Elephant (94.9), UCSB (89.5); worse on Fox (64.1 vs MILES 73.8). Baselines copied from earlier papers.
- *UCI small datasets (75).* Average-rank difference −3.92 vs SVM −3.23 (p = 0.15), SNN −2.85 (p = 0.10), random forest −2.79 (p = 0.05): the lead over the top three is not significant. Hyperparameters were grid-searched per dataset; baseline scores were taken from Klambauer et al., not re-run.
- *Drug design (50 random splits, baselines from Jiang et al.).* Best on BACE (0.902) and SIDER (0.672); behind GCN/RF/XGBoost on HIV and BBBP.

**Ablations.** None isolating the Hopfield mechanism: no ablation of β, update steps or memory size. The only controlled comparisons are across method families.

**Strongest evidence.** The attention equivalence (Eq. 10) and the fixed-point characterisation. The empirical wins are mostly on small or structured tabular/MIL tasks; none involves language modelling. The paper reports no seeds, so the empirical results carry only the cross-validation or split spreads quoted.

**Failure modes and limits.** (1) Capacity and retrieval theorems assume random or well-separated patterns with small basins; real text embeddings are correlated, and the theory then predicts averaging over many patterns (global or large metastable states). (2) Retrieval is read-only: the "memory" is data supplied by the caller or learned by SGD. (3) No test-time writing and no forgetting experiment. (4) No LM perplexity or generation evidence at all. (5) No energy, hardware-efficiency or causal brain claim is made, so none can be borrowed.

**Falsifiable BrainSwap follow-up.**
*Hypothesis:* the bank64 perplexity gap is explained by sharpness (β, key norms) and key separation in a 64-dimensional head, not by "memory". *Prediction:* at matched initialisation and fixed β, `bank0` equals `attn`, and extra slots add no benefit beyond the same parameters spent on a wider FFN or extra prefix tokens; perplexity vs bank size (16/64/256) is flat or worsens. Diagnostics that separate mechanism from tuning: slot-key separation Δ against ln N / β, effective βM² after normalisation, and the share of softmax mass (or count k) landing on bank slots, per layer. *Falsifier:* with a margin and a seed count fixed before running, `bank64` at matched β/init beats `bank0` on **test** perplexity with a paired-seed confidence interval excluding zero, and the parameter-matched wider-FFN baseline does not close the gap.
*Strongest conventional baseline:* the same decoder with +131,072 parameters spent in FFN width, plus learned prefix ("register") tokens. Titans' own ablation shows persistent tokens help only slightly (§3), so a wide FFN is the honest comparator.

## 3. Anchor 2: Behrouz, Zhong, Mirrokni, Titans (arXiv 2501.00663v1)

**Biological mechanism vs algorithm.** Brain language is motivational: short-term vs long-term vs persistent memory, "surprising events are memorable", forgetting. The PDF has no hippocampus, replay or sleep content (zero hits for each). The algorithm is an online-learning inner loop: a small MLP `M` is trained by gradient descent on an associative loss **at inference time**, while the surrounding network's weights stay frozen.

**Equations, in words.**
- Key/value/query: linear projections of the token.
- Loss (Eq. 12): squared error between `M(k_t)` and `v_t`, i.e. learn to map keys to values.
- Surprise memory (Eqs. 13–14): memory = (1 − α_t)·previous memory + S_t; S_t = η_t·S_{t−1} − θ_t·∇ℓ(M_{t−1}; x_t). This is SGD with **momentum** (η, "past surprise"), **weight decay** (α, forgetting) and **learning rate** (θ, "momentary surprise"), all data-dependent. The prose says surprise is the gradient "with respect to the input", but the equations differentiate the loss with respect to the memory parameters; the equations are the operative definition.
- Read (Eq. 15): a forward pass of `M` with no update, `y = M*(q)`.
- Parallel training (Eqs. 16–18): the sequence is cut into chunks; gradients inside a chunk are evaluated at the chunk's starting weights, so within-chunk writes are mini-batched rather than truly sequential. Momentum is a linear recurrence solved by associative scan. In the experiments, α, θ, η depend on the token.
- Persistent memory: N_p learnable, input-independent tokens prefixed to the sequence; they act like a data-free key/value block (the paper ties this to FFN-as-attention and attention-sink arguments).
- Three wirings: **MAC** (retrieve from `M` with the current segment as query, concatenate persistent + retrieved + segment, run attention, then write the attention output back into `M`), **MAG** (memory and sliding-window attention in parallel, gated), **MAL** (memory layer before sliding-window attention). A memory-only variant is called LMM. Figures 2–5 mark the test-time roles: persistent tokens fixed, attention doing in-context learning, and the long-term memory still learning.
- Theorem 4.1 claims expressivity beyond TC⁰; no proof is given in this version.

**What changes in an LLM.** Add a second, **writable** store inside each block: its parameters change per sequence, with a decay gate. Attention stays short-range (window or segment) and acts as the "short-term" component. Training needs the inner-loop gradient inside the forward pass, made practical by chunking.

**Experiments and compute.**
- Models: 170M/340M/400M/760M parameters; the first three trained on 15B tokens, the 760M on 30B tokens, FineWeb-Edu, Llama-2 tokenizer (32K vocab), 4K training length, AdamW, lr 4e-4, cosine schedule, 0.5M-token batches, weight decay 0.1. **No GPU/TPU hours, FLOPs, hardware or energy are reported.** Fig. 9 throughput (≈, tokens/s, 2K–16K): Mamba2 ~42k, Titans MAL ~40.5k, Gated DeltaNet ~40k, MAC/MAG ~36k, LMM ~35k, Mamba ~34.5k, all flat in length; Transformer++ falls from ~50k to ~24k. The text's "MAL is faster than baselines" holds against Mamba and long-context Transformer++ only; MAL is level with Gated DeltaNet and below Mamba2, and LMM is about 15% slower than both.
- Language modelling and commonsense (Table 1). At 340M/15B tokens: Titans (LMM) average 46.17, MAC 47.36, MAG 47.54, vs Gated DeltaNet 45.42, TTT 44.51, Transformer++ 42.92. At 760M/30B: MAC 52.51, MAG 52.50, LMM 51.56, vs Gated DeltaNet-H2 51.49, Samba 51.08, Transformer++ 48.69. Margins over the best hybrid are about one accuracy point, with no seeds or intervals. The win is not uniform: at 400M, MAC's Wikitext perplexity (25.61) is worse than Samba (25.32) and Gated DeltaNet-H2 (24.19). The 400M baseline numbers are copied from the Gated DeltaNet paper, not re-run. One 760M Titans (MAL) SIQA entry (30.98) looks like a transcription error.
- Needle-in-haystack (Table 2, RULER S-NIAH, 2K–16K tokens). At 16K: Titans (MAC) 98.4 / 97.4 / 95.2 on the three variants vs TTT 88.4 / 4.4 / 0.0, Mamba2 5.4 / 0 / 0, DeltaNet 71.4 / 5.4 / 0. Only recurrent baselines are in this table; there is no Transformer row to compare in this table.
- BABILong (Fig. 6). Few-shot: Titans (MAC) compared with large pretrained models; the baselines end near 10⁵ tokens while Titans is plotted to ~10⁶ (≈53%). Fine-tuning: a fine-tuned MAC against fine-tuned small models, RAG over Llama-3.1-8B, and large models whose numbers come from the benchmark authors. Where curves overlap, the comparison is closer than the caption suggests: fine-tuned Mamba is ≈99% to ~3×10⁴ tokens and ≈92% at ~1.3×10⁵, against Titans at ≈97–99%. Beyond ~10⁵ only Titans (≈94% at 10⁶, ≈69% at 10⁷), RMT (≈34% at 10⁷) and the RAG baseline (≈37%) are plotted. The abstract's ">2M context" is attributed to needle-in-haystack tasks, but S-NIAH stops at 16K; the long-length evidence is this BABILong figure alone. Titans is fine-tuned on the task while the large models are few-shot, so those comparisons are asymmetric.
- Also: memory-depth study (Fig. 7: Pile subset, LMM with 1–4 memory layers vs Mamba only, 2K–32K tokens, 170M/360M/760M; going from one to four memory layers lowers perplexity by about 0.4 at 8K+ tokens for 360M and 760M, and Fig. 8 shows throughput falling from ≈45k to ≈24k tokens/s from one to four layers; a different setup from Table 1), time-series forecasting (Table 3), DNA classification (Table 4, competitive but not uniformly best: the neural memory trails HyenaDNA on Human Regulatory and Human OCR).
- The footnote says larger-model results are pending; the conclusion promises code "soon". Neither anchor model release was available in v1.

**Ablations (Table 5).** The "language modelling ppl" column equals the mean of Wikitext and LMB perplexity at 400M (checked: (25.03 + 28.99)/2 = 27.01), the accuracy column is the Table 1 average, and "long context" is a BABILong-style score. Starting from LMM (27.01 ppl, 47.83 acc, 92.68 long): no weight decay → 29.04 / 45.11 / 85.60; no momentum → 28.98 / 45.49 / 87.12; linear instead of deep memory → 28.49 / 46.97 / 85.34; no convolution → 28.73 / 45.82 / 90.28; **no persistent memory → 27.63 / 46.35 / 92.49**. Adding attention (MAC/MAG/MAL) improves all three columns. Forgetting and momentum matter most; persistent memory matters least. No variance is reported.

**Strongest evidence.** The ablation ordering (decay and momentum matter) and the long-context S-NIAH gap against other recurrent models at 8K–16K. These are consistent with the paper's own mechanism story.

**Failure modes and limits.** (1) Gains over strong hybrids are small and unreplicated, with copied baselines. (2) The needle tasks measure retrieval of a planted string, a task closer to a key-value store than to episodic abstraction. (3) The brain-inspiration is analogy: surprise-gated writing is not shown to match any human-memory phenomenon; the paper never tests an alternative write rule that is "less brain-like" but algorithmically equal beyond the delta-rule/momentum comparisons. (4) No energy claim, no causal claim about biology. (5) Unreleased code and missing compute and hardware details make the efficiency story unauditable. (6) The ">2M context" claim rests on one fine-tuned BABILong curve; the needle-in-haystack table stops at 16K.

**Falsifiable BrainSwap follow-up.**
*Hypothesis:* online, decaying, gradient-written memory improves perplexity mainly when information must cross a context boundary. BrainSwap's decoder truncates documents to 128 tokens. Such windows can still benefit from the inductive bias of a memory module, but they do not test retention across windows; that test must stream related documents (e.g. consecutive 20 Newsgroups posts from one group) without resetting the memory. *Prediction:* a writable memory with decay lowers test perplexity on later posts relative to the same model with memory reset per document, and the benefit shrinks if decay is set to zero on long streams (stale associations interfere) or if the stream is shuffled across groups. *Falsifier:* perplexity with a persistent, decaying memory is no better than reset, or is matched by a non-parametric cache.
*Strongest conventional baseline:* a cache/kNN-LM-style retrieval over past hidden states (§6) and plain longer-context attention over the same stream, both at equal parameter count. A TTT-style inner loop without momentum and decay is the matched ablation.

## 4. Anchor 3: McClelland, McNaughton, O'Reilly (1995), complementary learning systems

**Biological mechanism vs algorithm.** This is the most biological of the three, but the 1995 paper is a **theory and a set of small simulations**, not a deployable algorithm. The mechanism: the neocortex learns slowly by small, interleaved weight changes that extract shared structure across experiences; the hippocampal system rapidly stores a compressed, sparse, conjunctive "summary sketch" of each episode using associative LTP; it reinstates patterns in the neocortex (during rehearsal or sleep) so that each reinstatement makes a small cortical change, until the memory no longer needs the hippocampus. The hippocampus is a teacher that supplies training trials; the cortex selects its own representations through interleaved learning.

Key evidence (clinical review): in the rat and monkey lesion studies, lesioned animals perform better the longer the delay between learning and lesion, and lesions within a day of the experience leave performance near chance in two of the studies. That rules out the simple alternative that every memory is stored twice, in a fast and a slow store that merely decay at different rates; it requires memory to change form or location over time. The human ECT data (TV-show test) support the same pattern but are confounded by depression at storage time, as the authors note.

Important gap between theory and simulation: in the consolidation simulations the hippocampus is a **black box** with a decaying trace strength and a reinstatement probability. No pattern-separation or pattern-completion network is simulated. The paper also argues that sparse conjunctive hippocampal coding serves to minimise interference among sequentially stored memories, so the fast store itself needs separated codes to survive months of consolidation; this parallels the key-separation condition in §2.

**Equations and arguments, in words.**
- *Why slow:* stochastic learning estimates population statistics, and a smaller learning rate averages over more samples; for gradient descent in multilayer nets, large steps break the assumed local linearity. A decaying cortical learning rate is optimal for online estimation, which the authors tie to developmental critical periods.
- *Catastrophic interference:* a net trained on AB pairs then on AC pairs loses AB almost entirely, unlike humans (McCloskey & Cohen); fast focused training of "penguins can swim, cannot fly" disturbs other birds, whereas interleaving it with the old corpus does not.
- *Why not make representations non-overlapping:* that removes interference but also removes shared-structure generalisation.
- *Two-store consolidation model (Eqs. 3–9):* hippocampal trace strength decays exponentially at rate D_h; the probability of a reinstatement per unit time is r_h times the current strength; cortical strength increases by C·S_h(t)·(1 − S_c) and decays at D_c, with C = (cortical learning rate) × r_h; behaviour is correct if either store succeeds, b_hc = b_h + (1 − b_h)·b_c, with b_h = R_h·S_h and b_c = R_c·S_c, plus a chance or prior-tendency term, b_t = b_hc + (1 − b_hc)·b_p.
- Two separate fits. The simplified two-store model (Fig. 1, Table 1) was fitted to four studies; Table 1 hippocampal decay D_h runs from 0.25 (Winocur) and 0.05 (Kim & Fanselow) through 0.035 (Zola-Morgan & Squire; C = 0.02, D_c = 0.003) to 0.001 (human ECT), with consolidation rate C covarying with D_h. The authors link fast decay to weak or passive experience and slow decay to salient experience, citing LTP decay rates of about 0.28 per day for weak and 0.05 per day for strong induction. The network simulations (Figs. 12–13) are a different fit: Zola-Morgan & Squire used cortical learning rate 0.03, hippocampal decay 0.025/day (about one sixth of strength left at 10 weeks), reinstatement 0.1 per training trial per day, test retrieval probability 0.07, a 50–15–1 network on 100 sparse random pairs plus 250 background items at p = 0.2 per day, 200 simulated subjects. The Kim & Fanselow simulation (16–16–16 network, 20 random background pairs once per day, one new pair) assumed no hippocampal decay and constant reinstatement, and the authors say other assumptions also fit.
- *Race between decay and consolidation:* if the hippocampal trace decays faster than cortex learns, idiosyncratic material is lost and what is shared across episodes survives, which they offer as an account of the schematic character of long-term memory.
- *Hippocampal indexing:* compression into and decompression out of the hippocampus relies on connections inside the slow neocortical system, so memory should be far better for material that fits previously learned structure. The authors also discuss Teyler & Discenna's index idea and leave open whether some fast learning sits in superficial cortex; they say they cannot choose between the two readings.

**What changes in an LLM.** Direct mapping: pretrained weights are the slow system; an episodic store (retrieval index, cache or fast weights) is the fast system; **replay** of stored items mixed with samples of old data is the consolidation mechanism; decay of the fast store is forgetting. This is a design principle for continual learning, not a drop-in layer.

**Experiments and compute.** Small connectionist simulations and four fits of the two-store model to published animal and human retrograde-amnesia data. No GPU, no energy, no scale beyond a few dozen units.
- Rumelhart semantic network (replication, learning rate 0.1, 500 epochs): representations differentiate coarse to fine (plants vs animals first, then subtypes), and a new concept (sparrow) trained on one relation inherits the others from similar concepts when interleaved with the corpus. The paper links this order to Keil's developmental data.
- AB–AC (Fig. 10): the network's AB list falls to zero within the first few AC trials, whereas humans retain roughly half; the paper calls this catastrophic interference.
- Penguin (Fig. 11): focused training on one new fact is learned faster but raises the error on the existing corpus sharply and cumulatively; interleaving with the full corpus is learned more slowly with almost no added interference.
- Retrograde-amnesia fits: Kim & Fanselow and Zola-Morgan & Squire as network simulations, plus the two-store fits to Winocur and Squire & Cohen. The authors describe the Zola-Morgan & Squire evidence for consolidation as relatively weak: the simulated lesion curve rises only slightly (≈0.60 to 0.69) partly because much cortical learning happens during the training sessions themselves.

**Ablations.** Focused vs interleaved training is the central controlled comparison. The consolidation fits vary decay, learning rate and reinstatement parameters, noting that sizeable changes worsen the fit. No variant of the architecture is ablated.

**Strongest evidence.** The qualitative two-system argument (interleaving prevents interference; slow learning discovers structure) and its ability to reproduce temporally graded retrograde amnesia. These are internally coherent and widely reused.

**Failure modes and limits.** (1) The hippocampus is not simulated; the claims about sparse conjunctive coding and attractor completion rest on cited physiology, not on a model here. (2) Fits use few parameters and noisy data, with time scales differing by more than two orders of magnitude between studies that the authors explain by speculation (age, species, stimulus strength); decay and consolidation rates are free per study, so the model does not predict a time scale. (3) "Interleaving prevents interference" is shown in tiny nets; its status for 10⁸–10¹¹ parameter LMs is an empirical question. (4) The paper itself says the analysis is incomplete. (5) It does not address energy or efficiency.

**Falsifiable BrainSwap follow-up.**
*Hypothesis:* hmem's null may reflect that its store adds little beyond the trained classifier (one candidate among those listed in §1). A fast store should pay when classes arrive sequentially and the slow learner has not seen them. *Prediction (class-incremental Banking77, classes revealed in groups):* a fast non-parametric store written at arrival lets the slow classifier retain old classes when it is trained with replay interleaved from that store; **focused** replay of only the newest items hurts old classes (the penguin result), and removing decay of the store hurts less than removing interleaving. *Falsifier:* focused and interleaved replay give the same old-class accuracy, or the store with no consolidation (pure kNN over stored items) matches the consolidated classifier.
*Strongest conventional baseline:* an experience-replay buffer (Rolnick et al.) with the same memory budget, bounded above by joint retraining on all data seen so far.

## 5. Synthesis by theme

| Theme | Hopfield (2020) | Titans (2024) | CLS (1995) |
|---|---|---|---|
| Fast weights | none; stored patterns are data or SGD-learned parameters | the inner loop *is* a fast-weight memory, trained by gradient steps at inference | fast learning is assigned to a separate hippocampal system, not to cortical weights |
| Associative memory | softmax read over stored patterns (key→value) | write = fit key→value mapping with squared error; read = forward pass | attractor completion in a sparse store (asserted, not simulated) |
| Indexing | stored patterns act as an index only if the user supplies them | none; memory is parametric, not an index | hippocampus holds a compressed sketch whose decoding depends on slow cortical weights |
| Replay / consolidation | none | none; no replay | central: reinstatement drives interleaved cortical updates |
| Forgetting | sketch only (recency penalty, unevaluated) | learned weight decay; ablation shows it matters | trace decay plus cortical decay; race with consolidation |
| Write/read separation | read-only | writes during the forward pass; reads without update | write in fast store; read, and teach the slow store, via reinstatement |

Distinctions that matter for BrainSwap:
- **Parametric slots** (bank64: written by SGD, static) vs **writable memory** (Titans' inner loop; Memorizing Transformers' key–value memory written at inference). Reporting either as "memory" without that distinction conflates them.
- **Read-only retrieval over fixed data** (hmem, and typically kNN-LM datastores and RAG corpora, which are built offline) separates what the model stores from what the retrieval corpus holds; there is no consolidation step unless a training loop reads from the store. Not every retrieval cache is written per test input.
- Only CLS predicts **when** a fast store should help (new, non-redundant material arriving over time). Only Titans supplies a concrete fast-write rule. Only Hopfield supplies a retrieval analysis, and it points to key separation and sharpness as the controlling quantities.
- None of the three supports an energy-efficiency or causal-brain claim.

**Main recommendation and the strongest objections.** Distinguish static associative memory from online writable memory, and test the latter against conventional baselines when sequential acquisition is the scientific target (§6). Static slots and labeled-set retrieval are legitimate memory mechanisms, but they do not establish the behavior of online writing or consolidation. The strongest counterarguments:
- *The mechanism evidence is thin.* Titans' margins over the best hybrids are about one point with no seeds and partly copied baselines; at overlapping lengths on BABILong a fine-tuned Mamba is nearly level with Titans; CLS rests on networks of a few dozen units and a black-box hippocampus.
- *Cheap baselines may capture most of the gain.* A kNN or cache read, longer attention, or a replay buffer is simpler, and the replay result is mature (Rolnick et al.), so a CLS-style experiment may add little beyond known replay behaviour.
- *The nulls may be tuning, not mechanism.* §2 implies that β, key norms and normalisation could explain bank64 and hmem as well as "no memory benefit" does; moving to a writable memory first would repeat the confound at higher cost.
- *Scale and cost.* An inner-loop memory adds compute and optimisation choices (chunk size, decay, momentum) that confound a small-scale comparison, and the 128-token setup cannot show cross-window gains.
- *Answer.* Run the cheap diagnostic first (key separation, softmax mass on slots, parameter-matched controls); build the writable-memory stream test only if static-slot behaviour is explained and the conventional baselines leave a gap.

## 6. Conventional baselines that any memory claim must beat

These were not read in full; summaries are from abstracts and are discovery-level only.

- **TTT layers** (Sun et al.): hidden state is a model updated by self-supervised steps at test time; compared at 125M–1.3B with a Transformer and Mamba. Closest published relative of Titans; also the matched ablation without momentum and decay.
- **kNN-LM** (Khandelwal et al.): interpolate a base LM with nearest neighbours over a datastore of past hidden states; read-only, no training of the retrieval side. The simplest strong baseline for any cache-style memory.
- **RETRO** (Borgeaud et al.): chunked cross-attention to retrieved passages from a very large corpus during training and inference.
- **RAG** (Lewis et al.): a retriever plus a generator trained jointly for knowledge-intensive tasks. Titans' BABILong comparison includes a RAG baseline (per the text).
- **Experience replay** (Rolnick et al.): replay buffers as a strong continual-learning baseline.
- **Wider FFN / prefix tokens** as parameter-matched controls for static slots.

## 7. Discovery bibliography (not read in full; identifiers checked against the arXiv API)

| Angle | Paper | ID | Role |
|---|---|---|---|
| Test-time training | Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* | arXiv 2407.04620 v4 | baseline for Titans |
| Fast weights | Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* | arXiv 2102.11174 v3 | delta-rule fast weights; the lineage Titans extends |
| Nonparametric memory | Khandelwal et al., *Generalization through Memorization: Nearest Neighbor Language Models* | arXiv 1911.00172 v2 | read-only cache baseline |
| Retrieval pretraining | Borgeaud et al., *Improving language models by retrieving from trillions of tokens* | arXiv 2112.04426 v3 | RETRO |
| Retrieval generation | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | arXiv 2005.11401 v4 | RAG |
| Hippocampal indexing | Gutiérrez et al., *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models* | arXiv 2405.14831 v3 | index built from a knowledge graph with personalised PageRank; the closest LLM instantiation of the indexing idea |
| Replay | Rolnick et al., *Experience Replay for Continual Learning* | arXiv 1811.11682 v2 | replay baseline; reinforcement-learning setting |
| Write/read split | Wu et al., *Memorizing Transformers* | arXiv 2203.08913 v1 | non-differentiable (key, value) memory written at inference and read by approximate kNN |

Cited inside the anchors but unread and unverified here: Teyler & Discenna (1986), Krotov & Hopfield (2016), Ba et al. (2016) fast weights, Sukhbaatar et al. all-attention. They are pointers only.

## 8. Remaining limits

- Reference lists of all three papers, and the raw data and baselines they cite, were not checked.
- Hopfield proofs: statements, assumptions and structure only; intermediate algebra was not re-derived. The dense ridge-plot arrays in Figs. A.4–A.5 were inspected only at low resolution, so their per-head values are not extracted.
- Plot-derived values (Titans Figs. 6–9, Hopfield Fig. A.4, CLS Figs. 12–13) are approximate.
- Titans v1 states the scale of Tables 2–5 only partly; the ablation scale is inferred from the arithmetic check in §3. Theorem 4.1 has no proof, and no code or hardware is described.
- Compute and energy: none of the three anchors reports usable numbers; the only compute figure is BERT-small at about four V100-days.
