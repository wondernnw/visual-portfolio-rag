#!/bin/bash
#SBATCH --job-name=visual_rag
#SBATCH --partition=a100ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/visual_rag_%j.out
#SBATCH --error=logs/visual_rag_%j.err

# Load Anaconda module
module load lang/Anaconda3/2024.06-1

# Activate conda environment with poppler
source $(conda info --base)/etc/profile.d/conda.sh
conda activate colpali_env

# Set environment variables
export HF_HOME="/lustre/project/ki-qarbs/nwang01/VisualRagPipeline/models"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"

# Disable network access (use local models only)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fix PATH to use conda python
export PATH="/home/nwang01/.conda/envs/colpali_env/bin:$PATH"

# Working directory
cd /lustre/project/ki-qarbs/nwang01/VisualRagPipeline

# Create logs directory if it doesn't exist
mkdir -p logs

# Run Visual RAG pipeline
/home/nwang01/.conda/envs/colpali_env/bin/python run_visual_rag.py \
    --pdf "/lustre/project/ki-qarbs/nwang01/VisualRagPipeline/doc/handbuch_portfolio.pdf" \
    --query "Was sind die Bewertungskriterien für das Portfolio?" \
    --top-k 3 \
    --reindex

echo "Job completed at $(date)"
