#!/bin/bash
#SBATCH --job-name=fdi_detection_69bus
#SBATCH --account=def-arashmoh
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/gkianfar/scratch/Amin/fdi_detection/logs/%x_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/fdi_detection/logs/%x_%j.err
#SBATCH --mail-user=your_email@domain.com
#SBATCH --mail-type=END,FAIL

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

source /home/gkianfar/scratch/Amin/ENV/bin/activate

# ── Project directory ──────────────────────────────────────────────────────
PROJECT_DIR="/home/gkianfar/scratch/Amin/CB/Cyber"
cd $PROJECT_DIR

mkdir -p logs results models

echo "============================================================"
echo "  Job: $SLURM_JOB_NAME  | ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME | CPUs: $SLURM_CPUS_PER_TASK"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

# ── Install dependencies (if needed, first run only) ──────────────────────
# Uncomment on first run:
# pip install torch torchvision --no-index        # from CC wheelhouse
# pip install scikit-learn numpy pandas openpyxl --no-index
# pip install transformers accelerate --no-index  # for LLM

# ── Download LLM (run once, then comment out) ──────────────────────────────
# Mistral-7B-Instruct: fits on 1× A100 40GB in fp16 (~14GB VRAM)
# huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
#     --local-dir /home/gkianfar/scratch/Amin/llm_cache/Mistral-7B-Instruct-v0.3
#
# Alternative (smaller, ~4GB, good for testing):
# huggingface-cli download microsoft/phi-3-mini-4k-instruct \
#     --local-dir /home/gkianfar/scratch/Amin/llm_cache/phi-3-mini-4k-instruct

# ── Run experiment ─────────────────────────────────────────────────────────
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

# ── Quick check job (no LLM, 50 days only) ────────────────────────────────
# To run a quick check before full experiment:
# python main_train.py --excel "$EXCEL_PATH" --scenario S1 --n_days 50 --epochs 50
