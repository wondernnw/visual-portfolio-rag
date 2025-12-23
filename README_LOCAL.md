# Visual RAG Pipeline (Local/Cluster Version)

A **Visual Retrieval-Augmented Generation (RAG)** system for evaluating German student portfolios. Uses ColPali for visual document retrieval and Qwen2-VL for multimodal answer generation.

> **Note**: This version uses local models for deployment on HPC clusters (e.g., MOGON-NHR). For the Colab version with Groq API, see [README_COLAB.md](README_COLAB.md).

## Overview

Unlike traditional text-based RAG that relies on OCR, this system treats PDF pages as **images** to preserve:
- Table structures
- Document layouts
- Formatting and visual elements
- Handwritten annotations

## Architecture

```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         ↓
┌─────────────────┐
│    ColPali      │  Visual embeddings per page
│   (Retriever)   │  MaxSim scoring
└────────┬────────┘
         ↓
┌─────────────────┐
│  Query + Top-K  │  Most relevant pages
│     Pages       │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Qwen2-VL 7B   │  Vision-Language Model
│   (Generator)   │  Reads page images
└────────┬────────┘
         ↓
┌─────────────────┐
│  German Answer  │
└─────────────────┘
```

## Models

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Retriever | [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) | 3B | Visual document retrieval |
| Base Model | [vidore/colpaligemma-3b-pt-448-base](https://huggingface.co/vidore/colpaligemma-3b-pt-448-base) | 3B | ColPali base model |
| Generator | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | 7B | Multimodal answer generation |

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended: 40GB+ VRAM)
- Poppler (for PDF processing)

## Installation

### Local Setup

```bash
# Create conda environment
conda create -n colpali_env python=3.11 poppler -c conda-forge
conda activate colpali_env

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install byaldi transformers accelerate qwen-vl-utils sentence-transformers pymupdf

# Download models
git lfs install
git clone https://huggingface.co/vidore/colpali-v1.3
git clone https://huggingface.co/vidore/colpaligemma-3b-pt-448-base
```

### Cluster Setup (MOGON-NHR)

```bash
# Load modules
module load lang/Anaconda3/2024.06-1

# Create environment
conda create -n colpali_env python=3.11 poppler -c conda-forge
conda activate colpali_env

# Install packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install byaldi transformers accelerate qwen-vl-utils sentence-transformers pymupdf

# Set environment variables
export HF_HOME="/path/to/models"
export HF_HUB_OFFLINE=1
```

## Usage

### Interactive Mode

Ask multiple questions without reloading models:

```bash
python run_visual_rag.py --pdf "doc/handbuch_portfolio.pdf" --top-k 5 -i
```

```
Frage: Was sind die Bewertungskriterien?
Answer: Die Bewertungskriterien umfassen 10 Bereiche (A bis J)...

Frage: Sind Fach und Klassenstufe benannt?
Answer: Ja, Fach, Klassenstufe und übergeordnetes Thema sind eindeutig benannt...

Frage: exit
Goodbye!
```

### Single Query

```bash
python run_visual_rag.py \
    --pdf "doc/handbuch_portfolio.pdf" \
    --query "Was sind die Bewertungskriterien?" \
    --top-k 3
```

### Reindex PDF

Required when using a new PDF or after changes:

```bash
python run_visual_rag.py \
    --pdf "doc/handbuch_portfolio.pdf" \
    --query "..." \
    --top-k 3 \
    --reindex
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pdf` | Path to PDF file | Required |
| `--query` | Question to ask | None |
| `--top-k` | Number of pages to retrieve | 3 |
| `--reindex` | Force rebuild index | False |
| `-i, --interactive` | Interactive mode | False |

## SLURM Job Submission

```bash
sbatch submit_visual_rag.slurm
```

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=visual_rag
#SBATCH --partition=a100ai
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

module load lang/Anaconda3/2024.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate colpali_env

export HF_HOME="/path/to/models"
export HF_HUB_OFFLINE=1

python run_visual_rag.py \
    --pdf "doc/handbuch_portfolio.pdf" \
    --query "Was sind die Bewertungskriterien?" \
    --top-k 3
```

## Project Structure

```
VisualRagPipeline/
├── README.md                # Main README (links to both)
├── README_LOCAL.md          # This file
├── README_COLAB.md          # Colab version docs
├── run_visual_rag.py        # Main Visual RAG script
├── submit_visual_rag.slurm  # SLURM job script
├── models/
│   └── hub/                 # HuggingFace model cache
│       ├── colpali-v1.3/
│       ├── models--vidore--colpaligemma-3b-pt-448-base/
│       └── models--Qwen--Qwen2-VL-7B-Instruct/
├── doc/                     # PDF documents
│   └── handbuch_portfolio.pdf
└── .byaldi/                 # ColPali index storage
    └── visual_doc_index/
```

## Troubleshooting

### "No NVIDIA driver found"

You're on a login node without GPU. Get a GPU node:

```bash
srun --partition=a100ai --gres=gpu:1 --time=02:00:00 --pty bash
```

### "Cannot connect to huggingface.co"

Set offline mode and ensure models are downloaded locally:

```bash
export HF_HUB_OFFLINE=1
export HF_HOME="/path/to/models"
```

### "No module named 'byaldi'"

Install in conda environment:

```bash
conda activate colpali_env
pip install byaldi
```

### Adapter model not loading

Update `adapter_config.json` to point to local base model path:

```json
{
  "base_model_name_or_path": "/path/to/models/hub/models--vidore--colpaligemma-3b-pt-448-base/snapshots/main"
}
```

## Performance

- **Indexing**: ~1-2 minutes for 28 pages
- **Retrieval**: <1 second per query
- **Generation**: ~10-20 seconds per answer
- **GPU Memory**: ~25GB (both models loaded)

## License

MIT License

## Acknowledgments

- [ColPali](https://github.com/illuin-tech/colpali) - Visual document retrieval
- [Byaldi](https://github.com/AnswerDotAI/byaldi) - RAG wrapper for ColPali
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) - Vision-Language Model
- MOGON-NHR Cluster - University of Mainz
