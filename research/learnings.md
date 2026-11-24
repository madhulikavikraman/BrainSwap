# Research learnings

These lessons synthesize the [findings](findings.md) and supporting reviews. They describe conclusions and methodological lessons from the literature review; they are not new experimental results for BrainSwap.

## Scientific lessons

### Memory needs an explicit definition

Static learned slots, retrieval from a fixed dataset, sequence-specific fast weights, and persistent episodic storage have different write rules and information access. BrainSwap's static-bank experiments do not test online acquisition or consolidation. A memory claim becomes clearer when it specifies what is stored, who writes it, when it changes, and when it resets. See the [memory review](memory-review.md).

### A biological interpretation can describe an existing operation

One-step modern Hopfield retrieval can be equivalent to scaled dot-product attention. That equivalence explains behavior but does not itself add a capability. An improvement needs a distinct operation, information source, learning rule, or computation budget, followed by a comparison with the closest conventional alternative.

### Writing and forgetting deserve separate experiments

TTT and Titans supply concrete mechanisms for changing sequence-specific memory. Their evidence motivates testing writes, decay, interference, and reset boundaries separately. Surprise can indicate useful novelty or noise; selecting surprising observations does not automatically select observations worth remembering. See the [learning-rule review](learning-review.md).

### Retention alone does not establish successful continual learning

A model can preserve old behavior by learning very little. Useful evaluation measures acquisition and retention together, including contradictory updates, repeated entities, and changing domains. Ordinary caches, nearest-neighbor retrieval, and replay are serious competitors to a more elaborate fast/slow system.

### Biological plausibility, model quality, and energy are separate outcomes

Local synaptic updates can require many rounds of network communication. Sparse events can reduce an operation estimate while adding state-management or routing cost. A spiking model's quality and measured energy depend on its implementation and hardware. Neither mathematical energy nor an assumed cost per operation is a measurement of electrical consumption. See the [spiking review](spiking-review.md).

### More computation is not always better

The HRM papers motivate recurrent refinement but also show why halting and evaluation behavior matter. A strong recurrent model does not establish that its two-timescale hierarchy causes the gain. Additional updates can alter an already correct answer. Comparisons need matched repeated block evaluations and separate tests of hierarchy, recurrence, supervision, and halting. See the [architecture review](architecture-review.md).

### A neutral result has several possible explanations

BrainSwap's memory results do not identify whether the limitation is the mechanism, representation quality, initialization, temperature, key separation, or training budget. Training on the same examples also does not imply that the model's weights have retained all their information. The existing controls and inexpensive retrieval diagnostics help distinguish these possibilities before a larger experiment.

## Lessons about reviewing research

### Reading coverage is part of the evidence

A downloaded PDF, an abstract summary, a complete technical reading, and an independently reproduced experiment are different levels of work. The reading ledger records versions and scope; discovery references and partially inspected papers remain separate from the core reviewed corpus. Reading a proof does not mean independently certifying it, and reading a bibliography does not mean validating every cited source.

### Primary-source checks matter more than confident summaries

The spiking review required corrections to quoted classification values and overly broad causal and hardware claims. SpikeGPT also contains a disagreement between its ablation table and adjacent prose. Those cases show why tables, captions, methods, and appendices need to be checked together, with unresolved inconsistencies recorded explicitly.

### Agreement among reviewers is not replication

Different models can repeat the same interpretation or mistake. Independent review is useful when it exposes assumptions or objections, but empirical claims still depend on the underlying sources and experiments. Focused follow-up on a skipped section or disputed value is more informative than another generic summary of the same paper.

### The closest conventional baseline clarifies the contribution

An online memory should compete with conventional adaptive state or retrieval; a consolidation mechanism with replay; a spiking representation with an analog and suitable low-precision model; a local learning rule with the same network trained by backpropagation. Matching information access, stored state, update work, and inference compute makes the resulting claim easier to interpret.
