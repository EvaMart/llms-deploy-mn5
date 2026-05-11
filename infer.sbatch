#!/bin/bash
#SBATCH --job-name=mn5_hello
#SBATCH --account=bsc619075
#SBATCH --qos=gp_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=infer_%j.out
#SBATCH --error=infer_%j.err

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
srun python local_infer.py --model_dir /gpfs/home/bsc/bsc619075/models/Qwen2.5-0.5B-Instruct --prompt "Hello from local. One sentence:"