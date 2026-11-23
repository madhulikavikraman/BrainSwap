#!/bin/bash
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --job-name=g27-p2-gen
#SBATCH --output=logs/%x-%A_%a.out
# Reproduces Siddu's bank64-vs-attn generation result on real 20 Newsgroups text, 3 seeds.
# 6 tasks = bank {0 (standard attention), 64} x seeds {27,28,29}. Early stop on val perplexity.
set -euo pipefail
source /project2/xiangren_1987/athota/brainswap-p2/env.sh
cd "$ROOT"

BANKS=(0 64); SEEDS=(27 28 29)
i=$SLURM_ARRAY_TASK_ID
BANK=${BANKS[$((i / 3))]}; SEED=${SEEDS[$((i % 3))]}
echo "task $i: bank_size=$BANK seed=$SEED"; nvidia-smi --query-gpu=name --format=csv,noheader

$PY src/train_generation.py --seed $SEED --bank_size $BANK \
    --data_dir data/20ng/ --results_base results/gen-20ng/
