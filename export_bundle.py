#!/usr/bin/env python3
"""Export a retrieval bundle from a portfolio PDF using ColPali.

Runs on the cluster GPU. Produces a JSON bundle that the Mac Streamlit
app can load for evaluation via Groq API.

Usage:
    python export_bundle.py --portfolio uploads/portfolio.pdf \
                            --criteria criteria.json \
                            --top-k 3 \
                            --output bundles/bundle.json
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime

import fitz  # pymupdf


def pdf_to_base64_images(pdf_path: str, dpi: int = 150) -> list[str]:
    """Convert every page of a PDF to a base64-encoded JPEG string."""
    doc = fitz.open(pdf_path)
    imgs: list[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        imgs.append(base64.b64encode(pix.tobytes("jpeg")).decode())
    doc.close()
    return imgs


def main():
    parser = argparse.ArgumentParser(description="Export ColPali retrieval bundle")
    parser.add_argument("--portfolio", required=True, help="Path to portfolio PDF")
    parser.add_argument("--criteria", default="criteria.json", help="Path to criteria JSON")
    parser.add_argument("--top-k", type=int, default=3, help="Pages per criterion")
    parser.add_argument("--output", default=None, help="Output bundle JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.portfolio):
        print(f"ERROR: Portfolio not found: {args.portfolio}")
        sys.exit(1)
    if not os.path.exists(args.criteria):
        print(f"ERROR: Criteria file not found: {args.criteria}")
        sys.exit(1)

    # Load criteria
    with open(args.criteria) as f:
        criteria = json.load(f)
    print(f"[export] Loaded {len(criteria)} criteria from {args.criteria}")

    # Convert portfolio pages to base64 images
    print(f"[export] Converting portfolio pages to images ...")
    page_images = pdf_to_base64_images(args.portfolio)
    n_pages = len(page_images)
    print(f"[export] Portfolio has {n_pages} pages")

    # Load ColPali and index portfolio
    from byaldi import RAGMultiModalModel

    hf_home = os.environ.get(
        "HF_HOME",
        "/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/models",
    )
    # Try local model first — prefer HF cache snapshot format (works with byaldi)
    model_path = "vidore/colpali-v1.3"
    candidates = [
        os.path.join(hf_home, "hub/models--vidore--colpali-v1.3/snapshots"),
        os.path.join(hf_home, "hub/colpali-v1.3"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            if candidate.endswith("snapshots"):
                snaps = os.listdir(candidate)
                if snaps:
                    model_path = os.path.join(candidate, snaps[0])
            else:
                model_path = candidate
            break

    print(f"[export] Loading ColPali from {model_path} ...")
    rag = RAGMultiModalModel.from_pretrained(model_path)
    print("[export] ColPali loaded.")

    index_name = "bundle_export_index"
    print(f"[export] Indexing portfolio ...")
    rag.index(
        input_path=args.portfolio,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=True,
    )
    print("[export] Portfolio indexed.")

    # Search each criterion
    print(f"[export] Searching {len(criteria)} criteria (top_k={args.top_k}) ...")
    hits_map: dict[str, list[dict]] = {}
    for c in criteria:
        k = c["kriterium"]
        hits = rag.search(k, k=args.top_k)
        hit_list = []
        for h in hits:
            page_num = h.page_num
            # page_num from byaldi is 1-indexed
            img_idx = page_num - 1
            b64 = page_images[img_idx] if 0 <= img_idx < n_pages else ""
            hit_list.append({
                "page_num": page_num,
                "score": round(h.score, 4),
                "base64": b64,
            })
        hits_map[k] = hit_list
        pages_str = ", ".join(f"S.{h['page_num']}" for h in hit_list)
        print(f"  {k[:60]}... -> [{pages_str}]")

    # Build bundle
    bundle = {
        "portfolio_name": os.path.basename(args.portfolio),
        "exported_at": datetime.now().isoformat(),
        "top_k": args.top_k,
        "n_pages": n_pages,
        "criteria": criteria,
        "hits_map": hits_map,
    }

    # Write output
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"bundles/bundle_{ts}.json"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(bundle, f, ensure_ascii=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[export] Bundle saved: {args.output} ({size_mb:.1f} MB)")
    print("[export] Done.")


if __name__ == "__main__":
    main()
