#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --account=bsc23
#SBATCH --qos=acc_debug
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --output=/gpfs/home/bsc/bsc619075/test_scripts/llm_Qwen3.6_27B_gpu_%j.out
#SBATCH --error=/gpfs/home/bsc/bsc619075/test_scripts/llm_Qwen3.6_27B_gpu_%j.err

# --- MN5 recommendation when using srun ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# --- Modules + venv ---
module purge
module load intel mkl impi hdf5 python/3.12.1

# Prevent leakage from user/global packages
export PYTHONNOUSERSITE=1

unset PYTHONPATH

# Activate venv
source /gpfs/home/bsc/bsc619075/envs/llm/bin/activate

# Diagnostics
echo "which python:"
which python
echo "which pip:"
which pip
python -m pip --version
python -c "
import sys, torch
print('sys.prefix:', sys.prefix)
print('sys.base_prefix:', sys.base_prefix)
print('torch file:', torch.__file__)
print('torch version:', torch.__version__)
"

# --- Working dir (recommended: not /gpfs/home) ---
RUN_DIR=/gpfs/scratch/bsc23/runs/${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# --- Put HF caches on local NVMe ---
export TMPDIR=/scratch/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
export HF_HOME="$TMPDIR/hf_home"
export TRANSFORMERS_CACHE="$TMPDIR/hf_transformers"
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"
export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1


# --- Sanity prints ---
echo "Node: $SLURMD_NODENAME"
nvidia-smi || true

python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"

MODEL_DIR="/gpfs/projects/bsc23/models/Qwen3.6-27B"

srun python /gpfs/home/bsc/bsc619075/test_scripts/local_infer.py \
  --model_dir "$MODEL_DIR" \
  --prompt "Hello from MN5 GPU job from " \
  --max_new_tokens 100