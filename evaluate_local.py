#!/usr/bin/env python3
"""Local all-in-one portfolio evaluation using ColPali + Qwen2-VL.

Runs entirely on the cluster GPU. No Groq API needed.

Pipeline:
  1. Load criteria from criteria.json
  2. (Optional) Convert checkliste PDF to base64 images
  3. ColPali indexes portfolio, searches each criterion (top-k pages)
  4. Free ColPali from GPU
  5. Load Qwen2-VL, evaluate each criterion
  6. Free Qwen2-VL, generate PDF report

Usage:
    python evaluate_local.py \
        --portfolio uploads/portfolios/motivation_adam.pdf \
        --checkliste uni_doc/checkliste_motivation.pdf \
        --criteria criteria.json \
        --top-k 3 \
        --output results/eval.pdf
"""

import argparse
import base64
import gc
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import fitz  # pymupdf
import torch
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from chat_app.prompts import build_evaluation_prompt
from chat_app.engine import parse_json_from_text
from chat_app.pdf_report import make_pdf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_HOME = os.environ.get(
    "HF_HOME",
    "/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/models",
)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
def find_local_colpali() -> str:
    """Find ColPali model in HF cache."""
    snapshots_dir = os.path.join(HF_HOME, "hub/models--vidore--colpali-v1.3/snapshots")
    if os.path.exists(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        if snaps:
            return os.path.join(snapshots_dir, snaps[0])
    return "vidore/colpali-v1.3"


def find_local_vlm() -> str:
    """Find Qwen2-VL model in HF cache."""
    cache = os.path.join(
        HF_HOME, "hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots"
    )
    if os.path.exists(cache):
        snaps = os.listdir(cache)
        if snaps:
            return os.path.join(cache, snaps[0])
    return "Qwen/Qwen2-VL-7B-Instruct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pdf_to_base64_images(pdf_path: str, dpi: int = 150) -> List[str]:
    """Convert every page of a PDF to a base64-encoded JPEG string."""
    doc = fitz.open(pdf_path)
    imgs: List[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        imgs.append(base64.b64encode(pix.tobytes("jpeg")).decode())
    doc.close()
    return imgs


def free_gpu():
    """Force free GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# ColPali retrieval
# ---------------------------------------------------------------------------
def run_colpali_retrieval(
    portfolio_path: str,
    criteria: List[Dict],
    top_k: int,
) -> Dict[str, List[Dict]]:
    """Index portfolio with ColPali and retrieve top-k pages per criterion.

    Returns hits_map: {kriterium: [{page_num, score, base64}, ...]}
    """
    print("\n" + "=" * 50)
    print("Phase 1: ColPali Retrieval")
    print("=" * 50)

    # Convert portfolio pages to images
    print("[ColPali] Converting portfolio to images ...")
    page_images = pdf_to_base64_images(portfolio_path)
    n_pages = len(page_images)
    print(f"[ColPali] Portfolio has {n_pages} pages")

    # Load ColPali
    model_path = find_local_colpali()
    print(f"[ColPali] Loading from {model_path} ...")
    rag = RAGMultiModalModel.from_pretrained(model_path)
    print("[ColPali] Ready.")

    # Index portfolio
    print("[ColPali] Indexing portfolio ...")
    rag.index(
        input_path=portfolio_path,
        index_name="eval_local_index",
        store_collection_with_index=True,
        overwrite=True,
    )
    print("[ColPali] Portfolio indexed.")

    # Search each criterion
    print(f"[ColPali] Searching {len(criteria)} criteria (top_k={top_k}) ...")
    hits_map: Dict[str, List[Dict]] = {}
    for c in criteria:
        k = c["kriterium"]
        hits = rag.search(k, k=top_k)
        hit_list = []
        for h in hits:
            page_num = h.page_num
            img_idx = page_num - 1  # byaldi is 1-indexed
            b64 = page_images[img_idx] if 0 <= img_idx < n_pages else ""
            hit_list.append({
                "page_num": page_num,
                "score": round(h.score, 4),
                "base64": b64,
            })
        hits_map[k] = hit_list
        pages_str = ", ".join(f"S.{h['page_num']}" for h in hit_list)
        print(f"  {k[:60]}... -> [{pages_str}]")

    # Free ColPali
    del rag
    free_gpu()
    print("[ColPali] GPU memory freed.")

    return hits_map


# ---------------------------------------------------------------------------
# Qwen2-VL evaluation
# ---------------------------------------------------------------------------
def run_qwen2vl_evaluation(
    criteria: List[Dict],
    hits_map: Dict[str, List[Dict]],
    checkliste_images: Optional[List[str]],
) -> List[Dict]:
    """Evaluate each criterion using Qwen2-VL.

    Returns list of evaluation dicts with punkte, kommentar, seiten_referenzen, zitat.
    """
    print("\n" + "=" * 50)
    print("Phase 2: Qwen2-VL Evaluation")
    print("=" * 50)

    # Load Qwen2-VL
    vlm_path = find_local_vlm()
    print(f"[Qwen2-VL] Loading from {vlm_path} ...")
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    proc = AutoProcessor.from_pretrained(vlm_path, local_files_only=True)
    print("[Qwen2-VL] Ready.")

    evaluations: List[Dict] = []

    for i, c in enumerate(criteria):
        kriterium = c["kriterium"]
        max_punkte = float(c["max_punkte"])
        anmerkung = c.get("anmerkung_auswerter", "")
        erklaerung = c.get("erklaerung", "")
        hits = hits_map.get(kriterium, [])

        print(f"\n  [{i+1}/{len(criteria)}] {kriterium[:60]}  (max {max_punkte})")

        # Build multimodal content: checkliste images + retrieved pages
        content: List[Dict] = []
        if checkliste_images:
            for img in checkliste_images:
                content.append(
                    {"type": "image", "image": f"data:image/jpeg;base64,{img}"}
                )

        page_nums = []
        for h in hits:
            content.append(
                {"type": "image", "image": f"data:image/jpeg;base64,{h['base64']}"}
            )
            page_nums.append(h["page_num"])
            print(f"    page {h['page_num']}  score={h['score']:.4f}")

        # Build evaluation prompt (reuses chat_app.prompts)
        prompt = build_evaluation_prompt(kriterium, max_punkte, anmerkung, erklaerung)
        content.append({"type": "text", "text": prompt})

        # Generate with Qwen2-VL
        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(msgs)
        inp = proc(
            text=[txt],
            images=img_in,
            videos=vid_in,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        torch.cuda.empty_cache()
        out = vlm.generate(
            **inp, max_new_tokens=1024, temperature=0.1, do_sample=False
        )
        gen = out[:, inp["input_ids"].shape[1]:]
        raw = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        del inp
        torch.cuda.empty_cache()

        # Parse JSON result
        parsed = parse_json_from_text(raw)

        if parsed and isinstance(parsed, dict):
            punkte = min(float(parsed.get("punkte", 0)), max_punkte)
            kommentar = parsed.get("kommentar", "")
            seiten_ref = parsed.get("seiten_referenzen", page_nums)
            zitat = parsed.get("zitat", "")
            if punkte >= max_punkte:
                kommentar = ""
        else:
            punkte = 0.0
            kommentar = "Automatische Bewertung fehlgeschlagen."
            seiten_ref = page_nums
            zitat = ""
            print(f"    WARNING -- could not parse: {raw[:120]}")

        evaluation = {
            "kriterium": kriterium,
            "max_punkte": max_punkte,
            "punkte": punkte,
            "kommentar": kommentar,
            "seiten_referenzen": seiten_ref,
            "zitat": zitat,
        }
        evaluations.append(evaluation)
        print(f"    => {punkte}/{max_punkte}  {kommentar[:80]}")

    # Free Qwen2-VL
    del vlm, proc
    free_gpu()
    print("\n[Qwen2-VL] GPU memory freed.")

    return evaluations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Local portfolio evaluation with ColPali + Qwen2-VL (cluster GPU)"
    )
    parser.add_argument(
        "--portfolio", required=True,
        help="Path to the portfolio PDF",
    )
    parser.add_argument(
        "--checkliste", default=None,
        help="Path to the checkliste PDF (optional, provides visual context)",
    )
    parser.add_argument(
        "--criteria", default="criteria.json",
        help="Path to criteria JSON (default: criteria.json)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Pages to retrieve per criterion (default: 3)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PDF path (default: results/evaluation_<timestamp>.pdf)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.portfolio):
        print(f"ERROR: Portfolio not found: {args.portfolio}")
        sys.exit(1)
    if not os.path.exists(args.criteria):
        print(f"ERROR: Criteria file not found: {args.criteria}")
        sys.exit(1)
    if args.checkliste and not os.path.exists(args.checkliste):
        print(f"ERROR: Checkliste not found: {args.checkliste}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/evaluation_{ts}.pdf"

    print("=" * 60)
    print("  Local Portfolio Evaluation (ColPali + Qwen2-VL)")
    print("=" * 60)
    print(f"  Portfolio  : {args.portfolio}")
    print(f"  Checkliste : {args.checkliste or '(none)'}")
    print(f"  Criteria   : {args.criteria}")
    print(f"  Top-K      : {args.top_k}")
    print(f"  Output     : {args.output}")
    print("=" * 60)

    # Load criteria
    with open(args.criteria) as f:
        criteria = json.load(f)
    print(f"\n[Config] Loaded {len(criteria)} criteria from {args.criteria}")

    # Convert checkliste to images (if provided)
    checkliste_images = None
    if args.checkliste:
        print(f"[Config] Converting checkliste to images ...")
        checkliste_images = pdf_to_base64_images(args.checkliste)
        print(f"[Config] Checkliste: {len(checkliste_images)} page(s)")

    # Phase 1: ColPali retrieval
    hits_map = run_colpali_retrieval(args.portfolio, criteria, args.top_k)

    # Phase 2: Qwen2-VL evaluation
    evaluations = run_qwen2vl_evaluation(criteria, hits_map, checkliste_images)

    # Phase 3: PDF report
    print("\n" + "=" * 50)
    print("Phase 3: Generating PDF Report")
    print("=" * 50)
    make_pdf(
        evaluations=evaluations,
        output_path=args.output,
        portfolio_name=os.path.basename(args.portfolio),
    )

    # Summary
    total = sum(e["punkte"] for e in evaluations)
    total_max = sum(e["max_punkte"] for e in evaluations)
    print(f"\n{'=' * 60}")
    print(f"  RESULT:  {total} / {total_max} Punkte")
    print(f"  Report:  {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
