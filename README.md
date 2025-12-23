# Visual RAG Pipeline

A **Visual Retrieval-Augmented Generation (RAG)** system for evaluating German student portfolios in teacher education.

## How It Works

```
PDF → Page Images → ColPali (Visual Retrieval) → MaxSim Search → Top-K Pages → VLM → Answer
```

![Architecture](doc/image.png)

The system processes PDF pages as **images** to preserve tables, layouts, charts, and handwritten annotations.

## Deployment Options

| Version | Generator | Setup | Guide |
|---------|-----------|-------|-------|
| **Colab** | Llama Vision (Groq API) | Simple, cloud-based | [README_COLAB.md](README_COLAB.md) |
| **Local/Cluster** | Qwen2-VL-7B | HPC deployment, offline | [README_LOCAL.md](README_LOCAL.md) |

## Quick Links

- **Colab Notebook**: [visual_rag_pipeline_colab.ipynb](visual_rag_pipeline_colab.ipynb)
- **Local Script**: [run_visual_rag.py](run_visual_rag.py)

## Models

| Component | Model | Download |
|-----------|-------|----------|
| Retriever | ColPali v1.3 | [HuggingFace](https://huggingface.co/vidore/colpali-v1.3) |
| Base Model | colpaligemma-3b-pt-448-base | [HuggingFace](https://huggingface.co/vidore/colpaligemma-3b-pt-448-base) |
| Generator (Colab) | Llama 4 Scout | via Groq API |
| Generator (Local) | Qwen2-VL-7B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |

> **Note**: Models not included in repo (~25GB). See [README_LOCAL.md](README_LOCAL.md) for download instructions.

## License

MIT License
