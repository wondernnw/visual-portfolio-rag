"""EvaluationEngine -- core logic for portfolio evaluation.

Loads a pre-computed retrieval bundle (from export_bundle.py on the cluster).
VLM backend (Groq API) handles all generation.
No ColPali or GPU needed on the Mac side.
"""

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import fitz  # pymupdf

from chat_app.prompts import build_evaluation_prompt
from chat_app.vlm_backends import VLMBackend


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    """A single retrieved page from the bundle — matches byaldi Result interface."""
    page_num: int
    score: float
    base64: str


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

    Works with pre-computed retrieval bundles (no GPU needed).
    VLM (Groq API) handles all generation.
    """

    def __init__(self, config, vlm: VLMBackend):
        self.config = config
        self.vlm = vlm

        # Cached state
        self.portfolio_name: Optional[str] = None
        self.checkliste_images: Optional[List[str]] = None
        self.criteria: Optional[List[Dict]] = None
        self.hits_map: Optional[Dict[str, List[PageResult]]] = None
        self.evaluations: Optional[List[Dict]] = None

    # ---- Bundle loading -----------------------------------------------------

    def load_bundle(self, path: str) -> Dict:
        """Load a pre-computed retrieval bundle from export_bundle.py.

        The bundle contains criteria, and per-criterion retrieved pages
        with base64 images and scores.

        Returns:
            {"status": "ok", "portfolio_name": str, "criteria_count": int, "criteria": [...]}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Bundle not found: {path}")

        with open(path) as f:
            bundle = json.load(f)

        self.portfolio_name = bundle.get("portfolio_name", "Portfolio")
        self.criteria = bundle["criteria"]

        # Reconstruct hits_map as PageResult objects
        raw_hits = bundle.get("hits_map", {})
        self.hits_map = {}
        for criterion_key, hit_list in raw_hits.items():
            self.hits_map[criterion_key] = [
                PageResult(
                    page_num=h["page_num"],
                    score=h["score"],
                    base64=h["base64"],
                )
                for h in hit_list
            ]

        print(f"[Engine] Bundle loaded: {self.portfolio_name}, {len(self.criteria)} criteria")
        return {
            "status": "ok",
            "portfolio_name": self.portfolio_name,
            "criteria_count": len(self.criteria),
            "criteria": self.criteria,
        }

    # ---- Checkliste loading (for display images in evaluation) --------------

    def load_checkliste(self, path: str) -> Dict:
        """Load checkliste PDF images for use in evaluation prompts.

        Returns:
            {"status": "ok", "pages": int}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkliste not found: {path}")

        self.checkliste_images = pdf_to_base64_images(path)
        print(f"[Engine] Checkliste loaded: {len(self.checkliste_images)} pages")
        return {"status": "ok", "pages": len(self.checkliste_images)}

    # ---- Public MCP-callable methods ----------------------------------------

    def evaluate_criterion(self, criterion_index: int) -> Dict:
        """Evaluate a single criterion by its index.

        Returns:
            {"status": "ok", "evaluation": {kriterium, max_punkte, punkte, kommentar, seiten_referenzen, zitat}}
        """
        if not self.criteria:
            return {"status": "error", "message": "No criteria loaded. Load a bundle first."}
        if not self.hits_map:
            return {"status": "error", "message": "No search results. Load a bundle first."}
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
            return {"status": "error", "message": "No criteria loaded. Load a bundle first."}
        if not self.hits_map:
            return {"status": "error", "message": "No search results. Load a bundle first."}

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
            portfolio_name=self.portfolio_name or "Portfolio",
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
