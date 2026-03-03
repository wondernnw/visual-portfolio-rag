# Visual Portfolio RAG

A **Visual Retrieval-Augmented Generation** system for evaluating student portfolios. Processes PDF pages as images (preserving tables, handwriting, layout) and evaluates them against a checklist using ColPali retrieval + Groq API generation.

## Architecture

```
Cluster (GPU)                          Mac (no GPU)
┌─────────────────────┐                ┌─────────────────────────┐
│  export_bundle.py   │   scp bundle   │  streamlit chat_app/    │
│  ColPali retrieval  │ ─────────────> │  Groq API evaluation    │
│  → bundle.json      │                │  → PDF report           │
└─────────────────────┘                └─────────────────────────┘
```

ColPali needs a GPU but Streamlit can't run on the HPC cluster (networking restrictions). So the work is split:

- **Cluster**: ColPali indexes the portfolio, searches each criterion, exports a JSON bundle with base64 page images
- **Mac**: Loads the bundle, evaluates each criterion via Groq API, generates a PDF report

## Setup

**Mac** (Streamlit app):
```bash
pip install -r requirements.txt
```

**Cluster** (bundle export):
```bash
pip install -r requirements-cluster.txt
```

## Usage

### 1. Export bundle on cluster

```bash
# Copy portfolio to cluster
scp portfolio.pdf cluster:~/PortfolioEvalTool/uploads/

# Run export job
ssh cluster
sbatch submit_export.slurm uploads/portfolio.pdf

# Copy bundle back to Mac
scp cluster:~/PortfolioEvalTool/bundles/bundle_*.json ./bundles/
```

Or run directly (interactive node with GPU):
```bash
python export_bundle.py --portfolio uploads/portfolio.pdf --criteria criteria.json --top-k 3 --output bundles/bundle.json
```

### 2. Evaluate on Mac

```bash
streamlit run chat_app/app.py
```

In the browser:
1. Enter Groq API key
2. Upload the bundle JSON
3. Optionally upload the checkliste PDF (adds context images)
4. Click **"Bewertung starten"**
5. Download the PDF report

## Files

| File | Description |
|------|-------------|
| `criteria.json` | 10 fixed evaluation criteria (8.5 max points) |
| `export_bundle.py` | Cluster CLI: ColPali retrieval → JSON bundle |
| `submit_export.slurm` | SLURM job script for bundle export |
| `chat_app/` | Streamlit app (bundle upload → Groq evaluation → PDF) |
| `requirements.txt` | Mac dependencies (no GPU) |
| `requirements-cluster.txt` | Cluster dependencies (byaldi, torch) |

## Models

| Component | Model | Where |
|-----------|-------|-------|
| Retriever | ColPali v1.3 | Cluster GPU |
| Generator | Llama 4 Scout | Groq API |

## License

MIT License
