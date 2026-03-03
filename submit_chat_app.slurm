#!/bin/bash
#SBATCH --job-name=portfolio_chat
#SBATCH --partition=a100ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chat_app_%j.out
#SBATCH --error=logs/chat_app_%j.err
#SBATCH --mail-user=nwang01@students.uni-mainz.de

# Load Anaconda module
module load lang/Anaconda3/2024.06-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate colpali_env

# Set environment variables
export HF_HOME="/lustre/project/ki-qarbs/nwang01/VisualRagPipeline/models"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"

# Use local models only (no internet on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fix PATH to use conda python
export PATH="/home/nwang01/.conda/envs/colpali_env/bin:$PATH"

# Working directory
cd /lustre/project/ki-qarbs/nwang01/VisualRagPipeline

# Create directories
mkdir -p logs results uploads

# Install chat app deps (first time only, skips if already installed)
pip install --quiet streamlit fastmcp python-dotenv 2>/dev/null

# Port setup
PORT=8501
TUNNEL_PORT=9501

# Start reverse SSH tunnel from compute node back to login node
# This bypasses the HTTP proxy that blocks direct TCP connections
ssh -f -N -o StrictHostKeyChecking=no -i ~/.ssh/cluster_internal -R ${TUNNEL_PORT}:localhost:${PORT} 10.81.2.141

# Print connection info
echo "============================================"
echo "  Streamlit running on: $(hostname):${PORT}"
echo "  Reverse tunnel: login21:${TUNNEL_PORT} -> localhost:${PORT}"
echo ""
echo "  Connect from your Mac with:"
echo "  ssh -N -L ${PORT}:localhost:${TUNNEL_PORT} mogon"
echo ""
echo "  Then open: http://localhost:${PORT}"
echo "============================================"

# Run Streamlit (headless, no browser)
/home/nwang01/.conda/envs/colpali_env/bin/python -m streamlit run chat_app/app.py \
    --server.port ${PORT} \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
