#!/usr/bin/env python3
"""
Visual RAG Pipeline: ColPali retrieval + Qwen2-VL generation.
Adapted from Colab version for MOGON-NHR cluster with local models.

Uses direct PDF indexing (requires poppler via conda).
"""
import os
import sys
import argparse
import base64
from io import BytesIO
from typing import Any, List

from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Configuration
HF_HOME = os.environ.get("HF_HOME", "/lustre/project/ki-qarbs/nwang01/VisualRagPipeline/models")
INDEX_NAME = "visual_doc_index"


def find_local_colpali():
    """Find local ColPali model path."""
    # Check for direct path first
    direct_path = os.path.join(HF_HOME, "hub/colpali-v1.3")
    if os.path.exists(direct_path):
        return direct_path
    # Check HF cache format
    cache_path = os.path.join(HF_HOME, "hub/models--vidore--colpali-v1.3/snapshots")
    if os.path.exists(cache_path):
        snapshots = os.listdir(cache_path)
        if snapshots:
            return os.path.join(cache_path, snapshots[0])
    return "vidore/colpali-v1.3"


def find_local_vlm():
    """Find local VLM model path."""
    vlm_path = os.path.join(HF_HOME, "hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots")
    if os.path.exists(vlm_path):
        snapshots = os.listdir(vlm_path)
        if snapshots:
            return os.path.join(vlm_path, snapshots[0])
    return "Qwen/Qwen2-VL-7B-Instruct"


class MultimodalRAG:
    """
    Visual RAG: PDF → page images → visual embeddings → retrieval → VLM reasoning
    """

    def __init__(self, keep_retriever=False):
        self.rag_engine = None
        self.vlm_model = None
        self.vlm_processor = None
        self.index_loaded = False
        self.keep_retriever = keep_retriever  # Don't free retriever in interactive mode

    def _load_retriever(self):
        if self.rag_engine is None:
            model_path = find_local_colpali()
            print(f"Loading ColPali Retriever ({model_path})...")
            self.rag_engine = RAGMultiModalModel.from_pretrained(
                model_path,
                device="cuda"
            )
            print("Retriever loaded.")

    def _free_retriever(self):
        """Free retriever from GPU to make room for VLM."""
        if self.rag_engine is not None:
            print("Freeing ColPali from GPU...")
            del self.rag_engine
            self.rag_engine = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("GPU memory freed.")

    def _load_vlm(self):
        if self.vlm_model is None:
            vlm_path = find_local_vlm()
            print(f"Loading Qwen2-VL ({vlm_path})...")
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                vlm_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
            self.vlm_processor = AutoProcessor.from_pretrained(vlm_path, local_files_only=True)
            print("VLM loaded.")

    def ingest_pdf(self, pdf_path: str, force_reindex: bool = False):
        """Index PDF directly with ColPali (requires poppler)."""
        self._load_retriever()

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Indexing PDF: {pdf_path}")
        self.rag_engine.index(
            input_path=pdf_path,
            index_name=INDEX_NAME,
            store_collection_with_index=True,  # Store images for retrieval
            overwrite=force_reindex
        )
        self.index_loaded = True
        print("Indexing complete.")

    def search(self, query: str, k: int = 1) -> List[Any]:
        """Search with ColPali (MaxSim on visual patches)."""
        # Reload retriever if it was freed
        if self.rag_engine is None:
            model_path = find_local_colpali()
            print(f"Reloading ColPali from index ({model_path})...")
            self.rag_engine = RAGMultiModalModel.from_index(
                INDEX_NAME,
                index_root=".byaldi"
            )
            self.index_loaded = True
        return self.rag_engine.search(query, k=k)

    def generate_answer(self, query: str, results: List[Any]) -> str:
        """Generate answer with Qwen2-VL using retrieved page images."""
        # Free retriever to make room for VLM (unless in interactive mode)
        if not self.keep_retriever:
            self._free_retriever()
        self._load_vlm()

        # Build multimodal content with images from ColPali results
        content = []

        print(f"Using {len(results)} pages:")
        for i, result in enumerate(results, start=1):
            page_num = result.page_num
            print(f"  {i}. Page {page_num} (score: {result.score:.4f})")

            # Use base64 image directly from ColPali result
            content.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{result.base64}"
            })

        # Optimized German prompt for portfolio evaluation
        prompt = (
            "Du bist ein erfahrener Dozent im Lehramtsbereich und bewertest Studierenden-Portfolios.\n"
            "Analysiere die bereitgestellten Dokumentseiten sorgfältig.\n\n"
            "Das PDF-Dokument besteht aus zwei Teilen:\n"
            "1. Die ersten 10 Seiten bilden ein Handbuch für die Bewertenden mit den Bewertungskriterien.\n"
            "2. Der zweite Teil ist das eigentliche Studierenden-Portfolio.\n\n"
            "Anweisungen:\n"
            "- Lies den Text auf den Bildern aufmerksam.\n"
            "- Beantworte die Frage basierend auf dem sichtbaren Inhalt.\n"
            "- Zitiere relevante Textstellen wenn möglich.\n"
            "- Wenn mehrere Seiten relevant sind, kombiniere die Informationen.\n"
            "- Antworte auf Deutsch, sachlich und strukturiert.\n\n"
            f"Frage: {query}\n\n"
            "Antwort:"
        )

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # Process and generate with Qwen2-VL
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Clear CUDA cache before generation
        torch.cuda.empty_cache()

        output_ids = self.vlm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,  # Low temperature for factual consistency
            do_sample=False   # Deterministic generation
        )

        # Only decode the generated tokens (exclude input)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Clear inputs from GPU
        del inputs
        torch.cuda.empty_cache()

        return output_text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Visual RAG: ColPali retrieval + Qwen2-VL generation"
    )
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--query", help="Question to ask (or use interactive mode)")
    parser.add_argument("--reindex", action="store_true", help="Force reindex")
    parser.add_argument("--top-k", type=int, default=3, help="Number of pages to retrieve")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode for multiple queries")
    args = parser.parse_args()

    # Keep retriever loaded in interactive mode (needs more GPU memory)
    rag = MultimodalRAG(keep_retriever=args.interactive)

    # Index if needed, otherwise load existing index
    if args.reindex or not os.path.exists(".byaldi/" + INDEX_NAME):
        rag.ingest_pdf(args.pdf, force_reindex=True)
    else:
        print("Loading existing index...")
        rag.rag_engine = RAGMultiModalModel.from_index(
            INDEX_NAME,
            index_root=".byaldi"
        )
        rag.index_loaded = True
        print("Index loaded.")

    def process_query(query):
        print(f"\nQuery: {query}")
        print("=" * 50)

        try:
            results = rag.search(query, k=args.top_k)

            if not results:
                print("No relevant pages found.")
                return

            print(f"\n=== ColPali Retrieval Results ===")
            for r in results:
                print(f"  Page {r.page_num}: score = {r.score:.4f}")
            print("=" * 35)

            answer = rag.generate_answer(query, results)
            print(f"\nAnswer:\n{answer}")
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    # Single query mode
    if args.query and not args.interactive:
        process_query(args.query)

    # Interactive mode
    elif args.interactive:
        print("\n" + "=" * 50)
        print("Interactive Mode - Type 'exit' to quit")
        print("=" * 50)

        while True:
            try:
                user_query = input("\nFrage: ").strip()
                if user_query.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                if not user_query:
                    continue
                process_query(user_query)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
