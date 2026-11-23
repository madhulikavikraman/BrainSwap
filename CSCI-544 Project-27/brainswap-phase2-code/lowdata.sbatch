#!/bin/bash
#SBATCH --account=xiangren_1987
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --job-name=g27-p2-lowdata
#SBATCH --output=logs/%x-%A_%a.out
# CPU only: these models are small and GPU hours are nearly used up.
# 12 tasks = {banking77, 20ng} x seeds {27,28,29} x n {10,25}; each task trains all 4 variants.
set -euo pipefail
source /project2/xiangren_1987/athota/brainswap-p2/env.sh
cd "$ROOT"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(banking77 20ng); SEEDS=(27 28 29); NS=(10 25)
i=$SLURM_ARRAY_TASK_ID
DS=${DATASETS[$((i / 6))]}; SEED=${SEEDS[$(((i / 2) % 3))]}; N=${NS[$((i % 2))]}
echo "task $i: dataset=$DS seed=$SEED per_class=$N"

$PY src/train_lowdata.py --seed $SEED --per_class $N --variant all \
    --data_dir data/$DS/ --results_base results/lowdata-$DS/
