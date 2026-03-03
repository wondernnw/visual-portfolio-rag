"""EvaluationEngine -- core logic for portfolio evaluation.

ColPali handles multimodal retrieval (always on cluster GPU).
VLM backend (Groq API or local Qwen2-VL) handles generation.
No model swapping needed when using Groq — ColPali stays loaded.
"""

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

import fitz  # pymupdf

from chat_app.config import AppConfig, find_local_colpali
from chat_app.prompts import CHECKLISTE_EXTRACTION_PROMPT, build_evaluation_prompt
from chat_app.vlm_backends import VLMBackend


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


def parse_json_from_text(text: str):
    """Best-effort JSON extraction from free-form VLM output."""
    # 1. Fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 2. Raw text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 3. Greedy match
    for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# EvaluationEngine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """Stateful engine for portfolio evaluation against a checklist.

    ColPali stays loaded on GPU for retrieval.
    VLM (Groq API) handles all generation — no swapping needed.
    """

    def __init__(self, config: AppConfig, vlm: VLMBackend):
        self.config = config
        self.vlm = vlm

        # ColPali retriever — loaded once, stays in GPU memory
        self._rag = None

        # Cached state
        self.portfolio_path: Optional[str] = None
        self.checkliste_path: Optional[str] = None
        self.checkliste_images: Optional[List[str]] = None
        self.criteria: Optional[List[Dict]] = None
        self.hits_map: Optional[Dict[str, List]] = None
        self.evaluations: Optional[List[Dict]] = None

    # ---- ColPali lifecycle -------------------------------------------------

    def _load_colpali(self):
        if self._rag is None:
            from byaldi import RAGMultiModalModel

            path = find_local_colpali()
            device = self.config.device
            print(f"[ColPali] Loading retriever from {path} on {device}")
            self._rag = RAGMultiModalModel.from_pretrained(path, device=device)
            print("[ColPali] Ready.")

    def _load_colpali_index(self):
        if self._rag is None:
            from byaldi import RAGMultiModalModel

            print("[ColPali] Loading from index ...")
            self._rag = RAGMultiModalModel.from_index(
                self.config.index_name, index_root=".byaldi"
            )
            print("[ColPali] Ready.")

    # ---- Public MCP-callable methods ---------------------------------------

    def index_portfolio(self, path: str, force: bool = False) -> Dict:
        """Index the portfolio PDF with ColPali for visual retrieval.

        Returns:
            {"status": "ok", "pages": <int>, "path": <str>}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Portfolio not found: {path}")

        self._load_colpali()
        self._rag.index(
            input_path=path,
            index_name=self.config.index_name,
            store_collection_with_index=True,
            overwrite=force or True,
        )
        self.portfolio_path = path

        doc = fitz.open(path)
        n_pages = len(doc)
        doc.close()

        print(f"[Engine] Portfolio indexed: {path} ({n_pages} pages)")
        return {"status": "ok", "pages": n_pages, "path": path}

    def index_checkliste(self, path: str) -> Dict:
        """Extract evaluation criteria from the checkliste PDF.

        Returns:
            {"status": "ok", "criteria_count": <int>, "criteria": [...]}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkliste not found: {path}")

        self.checkliste_path = path
        self.checkliste_images = pdf_to_base64_images(path)

        content = [
            {"type": "image", "image": f"data:image/jpeg;base64,{img}"}
            for img in self.checkliste_images
        ]
        content.append({"type": "text", "text": CHECKLISTE_EXTRACTION_PROMPT})

        raw = self.vlm.generate(content, max_tok=4096)
        criteria = parse_json_from_text(raw)

        if not criteria or not isinstance(criteria, list):
            print(f"  WARNING -- raw VLM output:\n{raw}")
            return {"status": "error", "criteria_count": 0, "criteria": [], "raw": raw}

        self.criteria = criteria
        print(f"[Engine] Extracted {len(criteria)} criteria")
        return {
            "status": "ok",
            "criteria_count": len(criteria),
            "criteria": criteria,
        }

    def search_portfolio(self, query: Optional[str] = None, top_k: Optional[int] = None) -> Dict:
        """Search portfolio with ColPali for pages relevant to criteria.

        Returns:
            {"status": "ok", "results": {<criterion>: [{"page": <int>, "score": <float>}, ...]}}
        """
        top_k = top_k or self.config.top_k
        self._load_colpali_index()

        if query:
            hits = self._rag.search(query, k=top_k)
            results = {query: [{"page": h.page_num, "score": h.score} for h in hits]}
            return {"status": "ok", "results": results}

        if not self.criteria:
            return {"status": "error", "message": "No criteria extracted yet. Run index_checkliste first."}

        results_map: Dict[str, List] = {}
        hits_map_raw: Dict[str, List] = {}
        for c in self.criteria:
            k = c["kriterium"]
            hits = self._rag.search(k, k=top_k)
            hits_map_raw[k] = hits
            results_map[k] = [{"page": h.page_num, "score": round(h.score, 4)} for h in hits]

        self.hits_map = hits_map_raw
        print(f"[Engine] Searched portfolio for {len(self.criteria)} criteria")
        return {"status": "ok", "results": results_map}

    def evaluate_criterion(self, criterion_index: int) -> Dict:
        """Evaluate a single criterion by its index.

        Returns:
            {"status": "ok", "evaluation": {kriterium, max_punkte, punkte, kommentar, seiten_referenzen, zitat}}
        """
        if not self.criteria:
            return {"status": "error", "message": "No criteria extracted yet."}
        if not self.hits_map:
            return {"status": "error", "message": "No search results. Run search_portfolio first."}
        if criterion_index < 0 or criterion_index >= len(self.criteria):
            return {"status": "error", "message": f"Invalid criterion index: {criterion_index}"}

        c = self.criteria[criterion_index]
        return self._evaluate_single(c)

    def evaluate_all(self) -> Dict:
        """Evaluate the portfolio against all criteria.

        Returns:
            {"status": "ok", "evaluations": [...], "total": <float>, "total_max": <float>}
        """
        if not self.criteria:
            return {"status": "error", "message": "No criteria extracted yet."}

        # Search if not done yet
        if not self.hits_map:
            self.search_portfolio()

        evaluations = []
        for c in self.criteria:
            ev = self._evaluate_single(c)
            if ev["status"] == "ok":
                evaluations.append(ev["evaluation"])
            else:
                evaluations.append({
                    "kriterium": c["kriterium"],
                    "max_punkte": float(c["max_punkte"]),
                    "punkte": 0.0,
                    "kommentar": "Bewertung fehlgeschlagen.",
                    "seiten_referenzen": [],
                    "zitat": "",
                })

        self.evaluations = evaluations
        total = sum(e["punkte"] for e in evaluations)
        total_max = sum(e["max_punkte"] for e in evaluations)
        print(f"[Engine] Evaluation complete: {total}/{total_max}")
        return {
            "status": "ok",
            "evaluations": evaluations,
            "total": total,
            "total_max": total_max,
        }

    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate a PDF evaluation report.

        Returns:
            {"status": "ok", "path": <str>, "total": <float>, "total_max": <float>}
        """
        if not self.evaluations:
            return {"status": "error", "message": "No evaluations yet. Run evaluate_all first."}

        if output_path is None:
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.config.results_dir, f"evaluation_{ts}.pdf")

        from chat_app.pdf_report import make_pdf

        make_pdf(
            evaluations=self.evaluations,
            output_path=output_path,
            portfolio_name=self.portfolio_path or "Portfolio",
        )

        total = sum(e["punkte"] for e in self.evaluations)
        total_max = sum(e["max_punkte"] for e in self.evaluations)
        return {"status": "ok", "path": output_path, "total": total, "total_max": total_max}

    # ---- Internal ----------------------------------------------------------

    def _evaluate_single(self, criterion: Dict) -> Dict:
        """Evaluate a single criterion against the portfolio."""
        kriterium = criterion["kriterium"]
        max_punkte = float(criterion["max_punkte"])
        anmerkung = criterion.get("anmerkung_auswerter", "")
        erklaerung = criterion.get("erklaerung", "")
        hits = self.hits_map.get(kriterium, []) if self.hits_map else []

        # Build multimodal content: checkliste + retrieved portfolio pages
        content = []
        if self.checkliste_images:
            content.extend(
                {"type": "image", "image": f"data:image/jpeg;base64,{img}"}
                for img in self.checkliste_images
            )

        page_nums = []
        for h in hits:
            content.append(
                {"type": "image", "image": f"data:image/jpeg;base64,{h.base64}"}
            )
            page_nums.append(h.page_num)

        prompt = build_evaluation_prompt(kriterium, max_punkte, anmerkung, erklaerung)
        content.append({"type": "text", "text": prompt})

        raw = self.vlm.generate(content, max_tok=1024)
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
            print(f"  WARNING -- could not parse: {raw[:120]}")

        evaluation = {
            "kriterium": kriterium,
            "max_punkte": max_punkte,
            "punkte": punkte,
            "kommentar": kommentar,
            "seiten_referenzen": seiten_ref,
            "zitat": zitat,
        }
        print(f"  => {punkte}/{max_punkte}  {kommentar[:80]}")
        return {"status": "ok", "evaluation": evaluation}
