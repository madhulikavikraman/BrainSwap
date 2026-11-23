#!/bin/bash
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --job-name=g27-p2-gen-ctrl
#SBATCH --output=logs/%x-%A_%a.out
# Completes the generation 2x2 on 20 Newsgroups: bank {0, 64} x beta {fixed, learned}.
#   already run:  attn (reference), bank0+beta, bank64
#   this job:     tasks 0-2  bank0        (Hopfield module, no bank, fixed beta)  <- grader's control
#                 tasks 3-5  bank64+beta  (bank and learned beta together)
#   then:         tasks 6-8  bank0+mhainit (bank0 with PyTorch attention's init)
#                 submit with: sbatch --array=6-8 gen_controls.sbatch
#                 These tell an init effect apart from an implementation effect.
# Same settings as gen.sbatch. Use the same GPU type as the earlier runs if you can
# (check the nvidia-smi line at the top of their logs), so timing stays comparable.
set -euo pipefail
source /project2/xiangren_1987/athota/brainswap-p2/env.sh
cd "$ROOT"

SEEDS=(27 28 29)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i % 3))]}
case $((i / 3)) in
  0) FLAGS="--bank_size 0 --force_hopfield" ;;
  1) FLAGS="--bank_size 64 --learn_beta" ;;
  2) FLAGS="--bank_size 0 --force_hopfield --hopfield_init mha" ;;
  *) echo "unknown task $i"; exit 1 ;;
esac
echo "task $i: $FLAGS seed=$SEED"; nvidia-smi --query-gpu=name --format=csv,noheader

$PY src/train_generation.py --seed $SEED $FLAGS \
    --data_dir data/20ng/ --results_base results/gen-20ng/
