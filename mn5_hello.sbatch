#!/bin/bash
#SBATCH --job-name=mn5_hello
#SBATCH --account=bsc23
#SBATCH --qos=gp_debug
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=hello_%j.out
#SBATCH --error=hello_%j.err

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

srun bash -lc 'hostname; date; whoami; python --version || true'