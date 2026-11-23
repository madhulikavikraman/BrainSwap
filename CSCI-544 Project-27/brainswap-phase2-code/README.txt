BrainSwap Phase 2 on CARC — run order
1. bash setup_phase2.sh                       (login node, ~5 min)
2. smoke test on a compute node               (see chat for the salloc line)
3. sbatch lowdata.sbatch && sbatch gen.sbatch
4. squeue -u athota                           (watch)
5. $PY summarize.py results                   (after jobs finish)
