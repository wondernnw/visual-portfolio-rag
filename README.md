# 👁️ Visual RAG PoC: Multimodal Retrieval with ColPali

> **A Proof of Concept for retrieval-augmented generation that "sees" documents instead of just reading text.**

## 📌 Overview
Standard RAG systems rely on OCR (Optical Character Recognition) to parse text, often failing when documents contain complex **tables, charts, or multi-column layouts**.

This project implements a **Visual RAG pipeline** that treats PDF pages as high-fidelity images. It allows the AI to "look" at a chart in a PDF and answer questions about it, preserving all spatial context.

## 🏗️ Architecture
This pipeline is divided into two stages: **Ingestion (Pre-production)** and **Inference (Production)**.

![Visual RAG Architecture](./image.png)

### How It Works (Step-by-Step)
1.  **Ingestion:** The system processes PDF documents, converting pages into screenshots.
2.  **Indexing (ColPali):** We use **ColPali** (ColBERT + PaliGemma) to create visual embeddings of the pages. Unlike text embeddings, these capture layout and visual data.
3.  **Retrieval:** When a user asks a question, the system finds the most visually relevant page.
4.  **Generation (VLM):** The retrieved page image is sent to a **Vision Language Model (Llama 3.2 Vision via Groq)** along with the user's query to generate an accurate answer.

## 🚀 Key Features
* **Zero OCR:** Bypasses text extraction errors entirely.
* **Multimodal Understanding:** Can answer questions like *"What is the trend in Figure 3?"* or *"Compare the values in the 'Revenue' column."*
* **High Performance:** Uses Groq's LPU inference engine for sub-second responses.

## 🛠️ Tech Stack
* **Language:** Python
* **Retriever:** `vidore/colpali-v1.3` (via Byaldi)
* **Generator:** Llama 3.2 Vision (Groq API)
* **Environment:** Google Colab / Jupyter Notebook
## 📦 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mustafa-ayyub/VisualRagPipeline.git
    cd VisualRagPipeline
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    You will need a [Groq API Key](https://console.groq.com/).
    ```bash
    export GROQ_API_KEY="gsk_..."
    ```

4.  **Run the Notebook:**
    Open `visual_rag_pipeline.ipynb` in Jupyter Lab or VS Code and run the cells.

## 📄 License
MIT License