# Next steps: selective writing and consolidation for LLM memory

## Objective

The [midterm](../docs/midterm-results.md) tested only read-only memory, and the [findings](findings.md) identify writing as the untested part of brain-inspired memory: what gets stored, when it changes, and how interference is controlled. This project tests two brain-inspired write rules on a frozen pretrained LM:

- **Surprise-gated episodic writes.** Store an observation only when the model failed to predict it. Entries decay with age, and corrections overwrite the entry they contradict.
- **Consolidation through replay.** Periodically train a small adapter from selected stored episodes, as the hippocampus is thought to train the neocortex ([complementary learning systems](memory-review.md#4-anchor-3-mcclelland-mcnaughton-oreilly-1995-complementary-learning-systems)).

### Contribution and prior work

Neither mechanism is new. Titans gates memory writes by surprise, and [SuRe](https://arxiv.org/abs/2511.22367) uses surprise-prioritised replay for continual LLM learning. The contribution is the **controlled comparison**: each brain-inspired rule against write-everything, random writes and plain retrieval, at an equal memory and replay budget, on a stream that contains corrections and noisy distractors. A negative result, where simple policies match the brain-inspired rule, is a valid outcome and gets reported as such.

## Rules for all plans

### Splits and selection

- Configurations are selected on validation streams only.
- Test streams are generated once in Plan 1, hashed, and not read until a plan's configuration is frozen.
- Each plan scores its frozen configuration on test exactly once and records the result.

### Budgets

Within a comparison, every policy gets the same:

- backbone
- store capacity (entries)
- retrieval `k`
- prompt length
- number of writes, where a policy controls that number
- replay tokens and adapter updates, in Plan 3

### Statistics

At least 3 seeds per configuration. Each seed gets fresh streams from the generator. Report the mean, per-seed paired differences against the named baseline, and 95% intervals.

### The CARC round trip

The agent has no CARC access. It pushes batch scripts to GitHub; the user runs them on CARC and pushes the results back. [Running on CARC](../docs/carc.md) defines the setup, the steps of each round trip and the job-script conventions.

Each batch covers a whole sweep, so that a plan needs only a few round trips. The agent picks configurations only from the sweep the current plan declares.

Limits:

- Batch scripts score validation only.
- Test scoring is a separate script, `scripts/test_<plan>.sh`. The agent hands it over once per plan, after the configuration is frozen.
- The agent stops at a plan boundary and waits for approval before starting the next plan.
- If a batch fails on CARC, the agent fixes it from the pushed Slurm logs. It does not resubmit blindly.

## Plan 1: Benchmark, baselines and the CARC round trip

**Outcome.** A fact-stream benchmark, baseline results with intervals, and a tested CARC round trip.

### Open prerequisites (user decisions)

- **Allocation.** Account, GPU partition and type, GPU hours available, and the GPU-hour cap per plan.
- **Team.** Agreement on this direction, since the midterm workstreams in `PHASE2_HANDOFF.md` (on `main`) are not continued here.

### Shared design

- **Backbone.** Pythia-160M (deduped), frozen, loaded with Hugging Face `transformers`. Its training data is public, so a check that the synthetic entities are absent from pretraining is meaningful.
- **Fact stream.** A seeded generator emits fictional entities with attributes ("The capital of Velmora is Drask.") as a sequence of events:
  - new fact
  - repetition
  - correction (a new value for an existing entity and attribute)
  - distractor: natural text from WikiText, plus high-surprise noise such as random strings, so that a surprise gate can be fooled
  - query: a paraphrased question with the answer withheld

  Validation and test use disjoint entities and disjoint question templates.
- **Memory.**
  - Store: up to `C` entries.
  - Key: mean-pooled final hidden state of the observation.
  - Value: the observation text.
  - Read: the top-`k` entries by cosine similarity to the query are prepended to the prompt.

  The read path stays fixed across policies, so only the write policy varies.
- **Scoring.** The answer is the candidate value with the highest log-probability among the values for that attribute. Metrics:
  - query accuracy
  - accuracy on corrected facts (latest value)
  - stale-answer rate (an earlier value chosen)
  - accuracy against fact age
  - retrieval precision@`k`
  - writes made
  - store occupancy
  - query latency

### Work

1. **Novelty check.** Search 2025–26 work on surprise-gated memory writes for frozen LMs, surprise or prioritised replay for LLMs, and LLM knowledge editing. Start from Titans, SuRe, EM-LLM, which segments episodes by surprise, and ROME/MEMIT, the conventional fact-update baselines. Record the closest papers and how this comparison differs in `research/prior-work.md`.
2. **Code layout.** `src/stream.py` (generator), `src/memory.py` (store, read, write policies), `src/evaluate.py` (scoring), `scripts/` (CARC env, sbatch, ledger collection), `tests/`.
3. **Baselines.**
   - no memory
   - full stream in context, as a reference for streams that fit Pythia's 2,048-token window
   - write-everything with first-in-first-out eviction at capacity `C`
   - random writes at capacity `C`

   Sweep `C` and stream length.
4. **CARC round trip.** `scripts/env.example.sh`, `scripts/setup.sh`, the batch and test script templates, and `scripts/collect.sh`, as specified in [Running on CARC](../docs/carc.md).

### Acceptance

- Generator tests pass:
  - the same seed gives the same stream
  - corrections refer to existing entity–attribute pairs
  - validation and test entities and templates are disjoint
- No-memory accuracy is near chance on fictional facts.
- Full-context accuracy is high on streams that fit the context window.
- All baselines are in the ledger with 3+ seeds and intervals.
- At least one batch ran on CARC and its results came back through GitHub without manual fixes.
- `research/prior-work.md` exists and supports the contribution statement above, or that statement is revised.

## Plan 2: Surprise-gated writes, decay and correction overwrites

**Outcome.** A yes/no answer, with intervals, to whether the brain-inspired write rule beats write-everything and random writes at equal capacity.

### Work

1. **Surprise.** Surprise is the backbone's mean token negative log-likelihood of an observation, given the current memory read. An observation the memory already explains is not surprising. A write happens when surprise exceeds a threshold, calibrated on validation so the expected writes fit the budget.
2. **Decay.** Each entry has a strength that decays with age and is restored when the entry is retrieved. The lowest-strength entry is evicted first.
3. **Correction overwrite.** When a new write's key matches an existing entry above a similarity threshold and its value differs, the new entry replaces the old one.
4. **Comparisons at equal capacity and equal writes.**
   - write-everything
   - random writes
   - raw surprise, without conditioning on the memory read
   - the full rule
   - the full rule without decay
   - the full rule without overwrite

### Acceptance

- Paired differences with intervals for each comparison, on corrected-fact accuracy, stale-answer rate and overall accuracy.
- The share of writes spent on noise distractors is reported for every policy.
- The configuration is frozen on validation, then scored once on test.

## Plan 3: Consolidation through replay

**Outcome.** A plot of learning new facts against keeping old ability, for each replay policy at an equal replay budget.

### Work

1. **Adapter.** A LoRA adapter on the frozen backbone. During a "sleep" phase every `S` stream events, it trains on episodes replayed from the Plan 2 store.
2. **Replay selection at equal replay tokens and adapter updates.**
   - no consolidation
   - uniform from the store
   - reservoir sampling over the whole stream
   - surprise-ranked, which is the SuRe-style comparison
   - naive fine-tuning on recent events only
3. **Evaluation.**
   - fact accuracy with memory reads disabled (is the fact in the weights?)
   - fact accuracy with reads enabled
   - retention of facts that were later evicted from the store
   - perplexity drift on held-out WikiText, which measures damage to general ability

### Acceptance

- Every policy is reported on both axes: acquisition (fact accuracy) and retention (old-fact accuracy and perplexity drift). Retention alone does not count as success.
- The configuration is frozen on validation, then scored once on test.

## Schedule to the presentation

Day 1 is 2026-11-25. The presentation is around 2026-12-02; confirm the exact date.

| Days | Work |
|---|---|
| 1–2 | Plan 1 |
| 3–4 | Plan 2 |
| 5–6 | Plan 3 |
| 7 | Figures and slides from frozen test results |

If a plan slips, the presentation uses the last completed plan's test results. Plan 3 is dropped before Plans 1–2 are cut short. Longer-term directions are in [ideas](ideas.md).
