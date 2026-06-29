#!/bin/bash
#SBATCH --job-name=fdi_detection_69bus
#SBATCH --account=def-arashmoh
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/gkianfar/scratch/Amin/CB/results/logs/%x_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/CB/results/logs/%x_%j.err


# ── Environment ────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

source /home/gkianfar/scratch/Amin/CB/ENV/bin/activate

# ── Project directory: your .py code lives in CB/Cyber/ ────────────────────
PROJECT_DIR="/home/gkianfar/scratch/Amin/CB/Cyber"
cd $PROJECT_DIR

# logs go inside Cyber/ ; results & models stay in the parent CB/ (per config.py)
mkdir -p /home/gkianfar/scratch/Amin/CB/results/logs
mkdir -p /home/gkianfar/scratch/Amin/CB/results
mkdir -p /home/gkianfar/scratch/Amin/CB/models

echo "============================================================"
echo "  Job: $SLURM_JOB_NAME  | ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME | CPUs: $SLURM_CPUS_PER_TASK"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

# ── Run experiment ─────────────────────────────────────────────────────────
# NOTE: the Excel filename has spaces, so it MUST stay quoted.
EXCEL_PATH="/home/gkianfar/scratch/Amin/CB/IEEE 69 bus.xlsx"

echo "Starting FDI detection experiment..."
python main_train.py \
    --excel   "$EXCEL_PATH" \
    --scenario both \
    --n_days  356 \
    --epochs  800 \
    --use_llm \
    --sensitivity \
    --seed 42

echo "Job completed: $(date)"
