#!/bin/bash
#SBATCH --job-name=eval_portfolio
#SBATCH --partition=a100ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_portfolio_%j.out
#SBATCH --error=logs/eval_portfolio_%j.err
#SBATCH --mail-user=nwang01@students.uni-mainz.de

# Load Anaconda module
module load lang/Anaconda3/2024.06-1

# Activate conda environment with poppler
source $(conda info --base)/etc/profile.d/conda.sh
conda activate colpali_env

# Set environment variables
export HF_HOME="/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/models"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"

# Disable network access (use local models only)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fix PATH to use conda python
export PATH="/home/nwang01/.conda/envs/colpali_env/bin:$PATH"

# Working directory
cd /lustre/project/ki-qarbs/nwang01/PortfolioEvalTool

# Create output directories
mkdir -p logs results

# Run Portfolio Evaluation Agent
/home/nwang01/.conda/envs/colpali_env/bin/python evaluate_portfolio.py \
    --portfolio "/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/doc/portfolio_motivation.pdf" \
    --checkliste "/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/doc/checkliste_motivation.pdf" \
    --output "results/evaluation_$(date +%Y%m%d_%H%M%S).pdf" \
    --top-k 3 \
    --reindex

echo "Job completed at $(date)"
