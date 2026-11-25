# BrainSwap: brain-inspired memory writing for LLMs

CSCI-544 (Applied NLP) course project, Group 27.

BrainSwap tests whether brain-inspired rules for **writing** memory help a frozen pretrained language model keep track of a changing stream of facts. The rules are surprise-gated writes, decay, correction overwrites and consolidation through replay. Each rule is compared against write-everything, random writes and plain retrieval at an equal memory and replay budget. The project does not propose a new mechanism. It runs the matched comparison and reports negative results when that's what it finds.

The midterm tested read-only Hopfield memory and found that it ties or trails ordinary attention. See the [midterm results](docs/midterm-results.md). The [literature review](research/findings.md) identifies writing and interference as the untested part.

## Repository layout

```
README.md
docs/
  carc.md                how experiments run on CARC: setup, round trip, job conventions
  midterm-results.md     midterm validation numbers that motivate the current work
  course/                proposal, midterm report and deck, Phase 2 report draft (docx)
research/
  next-steps.md          the current plans (Plans 1–3), rules and schedule
  ideas.md               unscheduled directions, each with its deciding control
  findings.md            literature synthesis and reading ledger
  learnings.md           scientific and review-method lessons
  *-review.md            topic reviews: memory, spiking, learning rules, architecture
  sources/               local paper cache (gitignored)
```

Code, results and CARC scripts are added by [Plan 1](research/next-steps.md#plan-1-benchmark-baselines-and-the-carc-round-trip) under `src/`, `scripts/`, `tests/` and `results/`.

The midterm code, run configs and logs are at commit [`1e5f89b`](https://github.com/madhulikavikraman/BrainSwap/tree/1e5f89b) and on `main`.
