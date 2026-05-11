#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --account=bsc23
#SBATCH --qos=acc_debug
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/home/bsc/bsc619075/test_scripts/llm_gpu_%j.out
#SBATCH --error=/gpfs/home/bsc/bsc619075/test_scripts/llm_gpu_%j.err

# --- MN5 recommendation when using srun ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# --- Modules + venv ---
module purge
module load intel mkl impi hdf5 python/3.12.1
source /gpfs/home/bsc/bsc619075/envs/llm/bin/activate

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

MODEL_DIR="/gpfs/home/bsc/bsc619075/models/Qwen2.5-0.5B-Instruct"

srun python /gpfs/home/bsc/bsc619075/test_scripts/local_infer.py \
  --model_dir "$MODEL_DIR" \
  --prompt "Hello from MN5 GPU job." \
  --max_new_tokens 30
