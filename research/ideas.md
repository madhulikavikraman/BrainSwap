# Ideas beyond the current plans

These directions come from the [findings](findings.md) and are not scheduled. Each one names the control that would decide it. The current work is in [next steps](next-steps.md).

| Idea | What it tests | Deciding control | Source |
|---|---|---|---|
| Associative read | Replace the top-`k` cosine read with Hopfield softmax retrieval (β · similarity) over the store | Same store and candidate access as kNN; report β against key separation, since retrieval needs β·Δ well above ln N | [Memory review §2](memory-review.md#2-anchor-1-ramsauer-et-al-modern-hopfield-networks-arxiv-200802217v3) |
| Pattern-separated keys | Sparse or projected keys, so similar facts interfere less | Same key dimension and storage; measure interference between near-duplicate entities | [Findings coverage map](findings.md#coverage-map) |
| Learned fast-weight module (TTT / delta rule) | Sequence-specific weights updated at test time, instead of an explicit store | Simple delta-rule and non-adaptive recurrence at matched state size; predict before updating on each token | [Learning-rule review](learning-review.md#ttt-layers-learnable-sequence-specific-plastic-state) |
| Midterm diagnostics | Explain the midterm nulls: slot attention mass, key norms and separation, matched β and init, bank size 16/64/256, parameter-matched wider FFN | `bank0` against `attn` at matched init; bank sweep against a wider FFN | [Memory review §2](memory-review.md#2-anchor-1-ramsauer-et-al-modern-hopfield-networks-arxiv-200802217v3) |
| Two-timescale recurrence and halting | Hierarchical recurrent refinement for reasoning | Ordinary recurrence at matched repeated block evaluations; halting isolated from hierarchy | [Architecture review](architecture-review.md) |
| Spiking efficiency | Event-based computation for energy | Analog and low-precision twins; measured device energy, not operation counts | [Spiking review](spiking-review.md) |
| Predictive coding or local learning | Training without backpropagation | The same network trained by backpropagation, at equal quality, compared on total update cost | [Learning-rule review](learning-review.md) |
