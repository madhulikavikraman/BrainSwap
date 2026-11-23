# Sourced by every script. Reuses Roshan's Python env (torch already installed there).
export ROOT=/project2/xiangren_1987/athota/brainswap-p2
export RENV=/project2/xiangren_1987/rshashid/group27-brainswap/env
export PY=$RENV/bin/python
export PYTHONPATH=$ROOT/pylib:${PYTHONPATH:-}   # extra packages we install ourselves, never into Roshan's env
export HF_HOME=$ROOT/.hf_cache                   # keep HuggingFace downloads out of your small home quota
# Login node limits threads: use 1 there, and the job's CPU count inside Slurm jobs
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
