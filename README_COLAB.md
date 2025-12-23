# Visual RAG Pipeline (Colab Version)

A **Visual Retrieval-Augmented Generation (RAG)** system for evaluating German student portfolios. Uses ColPali for visual document retrieval and Llama Vision via Groq API for answer generation.

> **Note**: This version uses Groq API for cloud-based inference. For local/cluster deployment with Qwen2-VL, see [README_LOCAL.md](README_LOCAL.md).

## Architecture

```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         ↓
┌─────────────────┐
│    ColPali      │  Multi-vector visual embeddings
│   (Retriever)   │  per page
└────────┬────────┘
         ↓
┌─────────────────┐
│  MaxSim Search  │  Maximum similarity scoring
│                 │  for retrieval
└────────┬────────┘
         ↓
┌─────────────────┐
│    Top-K Pages  │  Most relevant pages
└────────┬────────┘
         ↓
┌─────────────────┐
│  Llama Vision   │  Via Groq API
│   (Generator)   │  Cloud inference
└────────┬────────┘
         ↓
┌─────────────────┐
│  German Answer  │
└─────────────────┘
```

## Methodology

1. **Ingest:** Convert PDF pages into screenshots
2. **Index:** Create visual embeddings using ColPali (multi-vector representations)
3. **Retrieve:** Find the most relevant page images via MaxSim similarity search
4. **Generate:** Pass page images + query to Llama Vision via Groq API

## Models

| Component | Model | Provider |
|-----------|-------|----------|
| Retriever | vidore/colpali-v1.3 | HuggingFace (local) |
| Generator | Llama 4 Scout 17B | Groq API (cloud) |

## Requirements

- Google Colab (with GPU runtime)
- Groq API Key

## Quick Start

### 1. Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/visual_rag_pipeline_colab.ipynb)

### 2. Set API Key

Add your Groq API key to Colab secrets:
1. Click the 🔑 icon in the left sidebar
2. Add secret: `GROQ_API_KEY` = `gsk_...`

### 3. Run All Cells

The notebook will:
1. Install dependencies (byaldi, pdf2image, groq)
2. Load ColPali retriever
3. Index your PDF document
4. Start interactive Q&A

## Usage

```python
# Initialize
rag_system = MultimodalRAG()
rag_system.authenticate()

# Index PDF
rag_system.ingest_pdf("path/to/document.pdf")

# Ask questions
results = rag_system.search("Was sind die Bewertungskriterien?", k=3)
answer = rag_system.generate_answer(query, results[0])
```

## Features

- **Cloud-based inference** - No local GPU required for generation
- **Fast responses** - Groq's LPU for sub-second inference
- **Visual understanding** - Reads tables, charts, layouts
- **Interactive mode** - Continuous Q&A loop

## Comparison: Colab vs Local Version

| Feature | Colab Version | Local Version |
|---------|---------------|---------------|
| Generator | Llama Vision (Groq) | Qwen2-VL-7B |
| Inference | Cloud API | Local GPU |
| GPU Required | Only for ColPali | Full pipeline |
| Setup | Simple | Complex |
| Cost | API usage | Hardware |
| Offline | No | Yes |

## File Structure

```
VisualRagPipeline/
├── README.md                         # Main README (links to both)
├── README_COLAB.md                   # This file
├── README_LOCAL.md                   # Local/cluster docs
├── visual_rag_pipeline_colab.ipynb   # Colab notebook
└── run_visual_rag.py                 # Local version script
```

## Troubleshooting

### "Invalid GROQ_API_KEY"

Ensure your API key:
- Starts with `gsk_`
- Is added to Colab secrets (not hardcoded)

### "No module named 'byaldi'"

Restart runtime after installation:
- Runtime → Restart runtime

### ColPali loading slow

First run downloads ~6GB model. Subsequent runs use cache.

## License

MIT License

## Acknowledgments

- [ColPali](https://github.com/illuin-tech/colpali) - Visual document retrieval
- [Byaldi](https://github.com/AnswerDotAI/byaldi) - RAG wrapper
- [Groq](https://groq.com/) - Fast LLM inference
- [Llama Vision](https://llama.meta.com/) - Multimodal model
