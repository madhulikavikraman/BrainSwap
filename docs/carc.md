# Running BrainSwap on CARC

USC's CARC cluster runs every experiment. The agent writing the code has no CARC access. It pushes job scripts to GitHub; the user runs them on CARC and pushes the results back. This document describes that round trip, the one-time setup, and the conventions every job script follows.

`scripts/` is created in [Plan 1](../research/next-steps.md#plan-1-benchmark-baselines-and-the-carc-round-trip). Until then, the commands below describe the intended interface.

## What is known about the cluster

These facts come from the group's midterm job scripts and logs, at commit [`1e5f89b`](https://github.com/madhulikavikraman/BrainSwap/tree/1e5f89b/CSCI-544%20Project-27/brainswap-phase2-code).

| Item | Value |
|---|---|
| Slurm account | `xiangren_1987` |
| CPU partition | `main` |
| GPU partition | `gpu`, requested with `--gres=gpu:1`; the midterm jobs got a Tesla P100 16 GB |
| Project storage | `/project2/xiangren_1987/<user>/`; the home directory quota is small |
| Internet | Available on login nodes only. Downloads (pip packages, Hugging Face models, datasets) happen during setup, not inside jobs. |
| Login nodes | Light work only. Thread counts are limited there; anything heavy goes through Slurm. |
| Python | The group shares an environment with PyTorch at `/project2/xiangren_1987/rshashid/group27-brainswap/env`. Extra packages go into a private `pylib/` with `pip install --target`, never into the shared environment. |

The midterm notes say the group's GPU hours were nearly used up. Check the remaining allocation before the first GPU batch.

## One-time setup (user, on CARC)

1. Log in to CARC.
2. Make sure `git push` to the GitHub repository works from CARC, with an SSH key or a token.
3. Clone the repository into project storage:
   ```bash
   cd /project2/xiangren_1987/<user>/
   git clone git@github.com:madhulikavikraman/BrainSwap.git brainswap
   cd brainswap && git checkout research/brain-inspired-llm-review
   ```
4. Copy `scripts/env.example.sh` to `scripts/env.sh` and fill in the placeholders: your `ROOT` and the Python path. `env.sh` is gitignored, so personal paths stay off GitHub.
5. Run the setup once, on the login node:
   ```bash
   bash scripts/setup.sh
   ```
   It checks that PyTorch imports and sees the right version. It installs `transformers`, `peft` and `datasets` into `pylib/`, and downloads Pythia-160M and WikiText into `$ROOT/.hf_cache`. It ends with a 1-minute CPU smoke test.

## Each round trip

| Step | Who | What |
|---|---|---|
| 1 | Agent | Pushes `scripts/batch_<plan>_<n>.sh` and states the command to run, the expected number of jobs, and the GPU hours it will use |
| 2 | User | `git pull && bash scripts/batch_<plan>_<n>.sh`, which submits the Slurm jobs and prints their ids |
| 3 | User | Waits until `squeue -u $USER` shows none of the batch's jobs |
| 4 | User | `bash scripts/collect.sh`, then `git add results && git commit -m "results: batch_<plan>_<n>" && git push` |
| 5 | Agent | Pulls, checks the ledger and logs, writes `results/status.md`, and prepares the next batch or a fix |

If a job fails, still run step 4. The agent diagnoses the failure from the pushed logs.

The test-set script, `scripts/test_<plan>.sh`, follows the same steps. The agent hands it over once per plan, after the plan's configuration is frozen on validation.

## Conventions for job scripts (agent)

- **Header.** Every job sets:
  - `--account=xiangren_1987`, and `--partition=gpu --gres=gpu:1` or `--partition=main`
  - explicit `--cpus-per-task`, `--mem` and `--time`
  - `--job-name=bs-<plan>-<batch>`
  - `--output=logs/%x-%A_%a.out`
- **Job arrays.** A batch is one or a few Slurm job arrays. The task id maps to a configuration through a config file the batch script writes, `results/batch_<plan>_<n>/configs.jsonl`, not through shell arithmetic.
- **Environment.** Jobs `source scripts/env.sh` and set `HF_HUB_OFFLINE=1`, because compute nodes have no internet. Thread variables come from `SLURM_CPUS_PER_TASK`.
- **First log line.** Each job prints its config, the GPU name (`nvidia-smi --query-gpu=name --format=csv,noheader`), and the git commit it runs.
- **Outputs.** Each run writes `results/batch_<plan>_<n>/<run_id>/summary.json` and `metrics.jsonl`. Checkpoints and adapters stay on CARC under `$ROOT/checkpoints/` and are never committed.
- **Cost.** The batch script prints the job count and requested GPU hours before submitting. A batch stays within the plan's GPU-hour cap.
- **Pre-flight.** Before pushing, the agent runs the same entry point on a tiny CPU configuration on its own machine.

## What `collect.sh` pushes

- `results/ledger.jsonl`, with one line per run: config, seed, metrics, Slurm job id, GPU name and git commit. Lines are added once per run, so rerunning `collect.sh` is safe.
- Each run's `summary.json` and `metrics.jsonl`.
- `logs/` for the batch's jobs, including failed ones.
- The `sacct` state of each job: completed, failed or timed out, with elapsed time.

Nothing larger than a few megabytes is committed.
