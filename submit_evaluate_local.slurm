#!/bin/bash
#SBATCH --job-name=eval_local
#SBATCH --partition=a100ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_local_%j.out
#SBATCH --error=logs/eval_local_%j.err
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

# Use local models only (no internet on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fix PATH to use conda python
export PATH="/home/nwang01/.conda/envs/colpali_env/bin:$PATH"

# Working directory
cd /lustre/project/ki-qarbs/nwang01/PortfolioEvalTool

# Create directories
mkdir -p logs results

# ---- Configure these paths before running ----
PORTFOLIO="${1:-uploads/portfolios/motivation_adam.pdf}"
CHECKLISTE="${2:-uni_doc/checkliste_motivation.pdf}"
CRITERIA="criteria.json"
TOP_K=3
OUTPUT="results/evaluation_$(date +%Y%m%d_%H%M%S).pdf"

echo "============================================"
echo "  Local Evaluation (ColPali + Qwen2-VL)"
echo "  Portfolio:   ${PORTFOLIO}"
echo "  Checkliste:  ${CHECKLISTE}"
echo "  Criteria:    ${CRITERIA}"
echo "  Top-K:       ${TOP_K}"
echo "  Output:      ${OUTPUT}"
echo "============================================"

# Build checkliste arg (skip if file doesn't exist)
CHECKLISTE_ARG=""
if [ -f "${CHECKLISTE}" ]; then
    CHECKLISTE_ARG="--checkliste ${CHECKLISTE}"
fi

/home/nwang01/.conda/envs/colpali_env/bin/python evaluate_local.py \
    --portfolio "${PORTFOLIO}" \
    ${CHECKLISTE_ARG} \
    --criteria "${CRITERIA}" \
    --top-k ${TOP_K} \
    --output "${OUTPUT}"

echo "Job completed at $(date)"
