#!/bin/bash
#SBATCH --job-name=export_bundle
#SBATCH --partition=a100ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/export_bundle_%j.out
#SBATCH --error=logs/export_bundle_%j.err
#SBATCH --mail-user=nwang01@students.uni-mainz.de

# Load Anaconda module
module load lang/Anaconda3/2024.06-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate colpali_env

# Set environment variables
export HF_HOME="/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/models"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"

# Note: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE removed — they block
# byaldi from loading local model files. No internet on compute nodes anyway.

# Fix PATH to use conda python
export PATH="/home/nwang01/.conda/envs/colpali_env/bin:$PATH"

# Working directory
cd /lustre/project/ki-qarbs/nwang01/PortfolioEvalTool

# Create directories
mkdir -p logs bundles

# ---- Configure these paths before running ----
PORTFOLIO="${1:-uploads/portfolios/motivation_adam.pdf}"
CRITERIA="criteria.json"
TOP_K=3
OUTPUT="bundles/bundle_$(date +%Y%m%d_%H%M%S).json"

echo "============================================"
echo "  Exporting retrieval bundle"
echo "  Portfolio: ${PORTFOLIO}"
echo "  Criteria:  ${CRITERIA}"
echo "  Top-K:     ${TOP_K}"
echo "  Output:    ${OUTPUT}"
echo "============================================"

/home/nwang01/.conda/envs/colpali_env/bin/python export_bundle.py \
    --portfolio "${PORTFOLIO}" \
    --criteria "${CRITERIA}" \
    --top-k ${TOP_K} \
    --output "${OUTPUT}"

echo "Job completed at $(date)"
