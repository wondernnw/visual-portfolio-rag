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
scp portfolio.pdf mogon-nhr:/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/uploads/portfolios/

# Run export job
ssh mogon-nhr
cd /lustre/project/ki-qarbs/nwang01/PortfolioEvalTool
sbatch submit_export.slurm uploads/portfolios/portfolio.pdf

# Check job status
squeue -u nwang01

# Copy bundle back to Mac
scp mogon-nhr:/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/bundles/bundle_*.json ./bundles/
```

Or run directly (interactive node with GPU):
```bash
python export_bundle.py --portfolio uploads/portfolios/portfolio.pdf --criteria criteria.json --top-k 3 --output bundles/bundle.json
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
5. View evaluation results in the chat
6. Download the PDF report from the sidebar

## How it works

### Cluster (`export_bundle.py`)
1. Loads `criteria.json` (10 fixed evaluation criteria, 8.5 max points)
2. ColPali indexes the portfolio PDF as page images
3. For each criterion, searches for the top-3 most relevant pages
4. Exports everything to a JSON bundle: criteria + page images (base64) + relevance scores

### Mac (`chat_app/`)
1. **Load bundle** -- reads JSON, populates criteria and retrieved pages
2. **Load checkliste** (optional) -- converts checkliste PDF to images for evaluation context
3. **Evaluate all** -- for each criterion, sends checkliste + portfolio pages + prompt to Groq API (Llama 4 Scout). Returns points, comment, page references, quote
4. **Generate report** -- builds a PDF table with scores, comments, and page references

### Two interaction modes
- **Button-driven** -- sidebar "Bewertung starten" runs the full pipeline automatically
- **Chat** -- free-text chat where the LLM decides which tools to call

## Files

| File | Description |
|------|-------------|
| `criteria.json` | 10 fixed evaluation criteria (8.5 max points) |
| `export_bundle.py` | Cluster CLI: ColPali retrieval -> JSON bundle |
| `submit_export.slurm` | SLURM job script for bundle export |
| `chat_app/app.py` | Streamlit entry point |
| `chat_app/engine.py` | Core logic: bundle loading, Groq evaluation |
| `chat_app/mcp_server.py` | FastMCP server with 5 tools |
| `chat_app/chat_agent.py` | LLM-driven + button-driven agents |
| `chat_app/vlm_backends.py` | Groq API wrapper |
| `chat_app/prompts.py` | German evaluation prompts |
| `chat_app/pdf_report.py` | PDF report generation |
| `requirements.txt` | Mac dependencies (no GPU) |
| `requirements-cluster.txt` | Cluster dependencies (byaldi, torch) |

## Models

| Component | Model | Where |
|-----------|-------|-------|
| Retriever | [ColPali v1.3](https://huggingface.co/vidore/colpali-v1.3) | Cluster GPU |
| Generator | [Llama 4 Scout 17B](https://groq.com/) | Groq API |

## The 10 Criteria

From "Motivationsfoerderliche Rueckmeldung" (8.5 max points):

1. Learning goals named (1.0)
2. Positive individual achievements described (1.0)
3. Individual learning progress described (1.0)
4. Effort appreciation (0.5)
5. Deficits precisely named (1.0)
6. Concrete remediation hints (1.0)
7. Grade/score provided (1.0)
8. Class ranking position (0.5)
9. Motivational closing remark (1.0)
10. Age-appropriate language (0.5)

## Cluster Troubleshooting

### "Cannot connect to huggingface.co"
Models must be pre-downloaded. Set offline mode:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/models"
```

### "Repo id must be in the form..."
The `adapter_config.json` in the ColPali model has a hardcoded path. Fix it:
```bash
cd /path/to/models/hub/colpali-v1.3
sed -i 's|OldProjectName|PortfolioEvalTool|g' adapter_config.json
```

Also fix paths in the HF cache snapshot:
```bash
cd /path/to/models/hub/models--vidore--colpali-v1.3/snapshots/<hash>/
sed -i 's|OldProjectName|PortfolioEvalTool|g' *.json
```

## License

MIT License
