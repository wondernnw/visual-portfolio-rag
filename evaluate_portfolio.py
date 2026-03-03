#!/usr/bin/env python3
"""
Portfolio Evaluation Agent
==========================
Evaluates student portfolios against a checklist using Visual RAG.

Pipeline:
  1. Index portfolio PDF with ColPali for visual retrieval
  2. Extract evaluation criteria from Checkliste via VLM
  3. Retrieve relevant portfolio pages per criterion (ColPali)
  4. Evaluate each criterion via VLM
  5. Generate structured PDF report

Output PDF columns:
  - Kriterien (Criteria)
  - Punkte (Points awarded / Max points)
  - Kommentar bei Punktabzug (Comment for deductions)
  - Motivationsfoerderliche Rueckmeldung (Section subtotal)
  - Gesamte Punkte (Total at the bottom)

Usage:
  python evaluate_portfolio.py \
    --portfolio uni_doc/portfolio_motivation.pdf \
    --checkliste uni_doc/checkliste_motivation.pdf \
    --output results/evaluation.pdf
"""

import os
import sys
import gc
import json
import argparse
import base64
import re
from datetime import datetime
from typing import Any, List, Dict, Optional

import fitz  # pymupdf – PDF to images
import torch
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fpdf import FPDF

try:
    from fpdf import FontFace
except ImportError:
    from fpdf.fonts import FontFace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_HOME = os.environ.get(
    "HF_HOME",
    "/lustre/project/ki-qarbs/nwang01/PortfolioEvalTool/models",
)
INDEX_NAME = "portfolio_eval_index"


# ---------------------------------------------------------------------------
# Model discovery (same logic as run_visual_rag.py)
# ---------------------------------------------------------------------------
def find_local_colpali() -> str:
    """Find ColPali model – local path or HuggingFace ID."""
    direct = os.path.join(HF_HOME, "hub/colpali-v1.3")
    if os.path.exists(direct):
        return direct
    cache = os.path.join(HF_HOME, "hub/models--vidore--colpali-v1.3/snapshots")
    if os.path.exists(cache):
        snaps = os.listdir(cache)
        if snaps:
            return os.path.join(cache, snaps[0])
    return "vidore/colpali-v1.3"


def find_local_vlm() -> str:
    """Find Qwen2-VL model – local path or HuggingFace ID."""
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
# EvaluationAgent
# ---------------------------------------------------------------------------
class EvaluationAgent:
    """End-to-end portfolio evaluation against a checklist."""

    def __init__(self):
        self.rag: Optional[RAGMultiModalModel] = None
        self.vlm = None
        self.proc = None

    # ---- GPU lifecycle -----------------------------------------------------

    def _load_colpali(self):
        if self.rag is None:
            p = find_local_colpali()
            print(f"[ColPali] Loading retriever from {p}")
            self.rag = RAGMultiModalModel.from_pretrained(p, device="cuda")
            print("[ColPali] Ready.")

    def _load_colpali_index(self):
        if self.rag is None:
            print("[ColPali] Loading from index ...")
            self.rag = RAGMultiModalModel.from_index(
                INDEX_NAME, index_root=".byaldi"
            )
            print("[ColPali] Ready.")

    def _free_colpali(self):
        if self.rag is not None:
            del self.rag
            self.rag = None
            torch.cuda.empty_cache()
            gc.collect()
            print("[ColPali] GPU memory freed.")

    def _load_vlm(self):
        if self.vlm is None:
            p = find_local_vlm()
            print(f"[Qwen2-VL] Loading VLM from {p}")
            self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                p,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
            )
            self.proc = AutoProcessor.from_pretrained(p, local_files_only=True)
            print("[Qwen2-VL] Ready.")

    def _free_vlm(self):
        if self.vlm is not None:
            del self.vlm, self.proc
            self.vlm = self.proc = None
            torch.cuda.empty_cache()
            gc.collect()
            print("[Qwen2-VL] GPU memory freed.")

    # ---- VLM generation helper ---------------------------------------------

    def _generate(self, content: List[Dict], max_tok: int = 1024) -> str:
        """Send multimodal content to Qwen2-VL and return generated text."""
        self._load_vlm()
        msgs = [{"role": "user", "content": content}]
        txt = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(msgs)
        inp = self.proc(
            text=[txt],
            images=img_in,
            videos=vid_in,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        torch.cuda.empty_cache()
        out = self.vlm.generate(
            **inp, max_new_tokens=max_tok, temperature=0.1, do_sample=False
        )
        gen = out[:, inp["input_ids"].shape[1]:]
        res = self.proc.batch_decode(gen, skip_special_tokens=True)[0]
        del inp
        torch.cuda.empty_cache()
        return res.strip()

    # ---- Pipeline phases ---------------------------------------------------

    def _phase1_index_portfolio(self, path: str, force: bool):
        """Index the portfolio PDF with ColPali."""
        print("\n" + "=" * 50)
        print("Phase 1: Indexing Portfolio")
        print("=" * 50)
        self._load_colpali()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Portfolio not found: {path}")
        self.rag.index(
            input_path=path,
            index_name=INDEX_NAME,
            store_collection_with_index=True,
            overwrite=force,
        )
        print("Portfolio indexed.")

    def _phase2_extract_criteria(self, cl_imgs: List[str]) -> List[Dict]:
        """Use VLM to extract evaluation criteria from the Checkliste."""
        print("\n" + "=" * 50)
        print("Phase 2: Extracting Criteria from Checkliste")
        print("=" * 50)
        self._free_colpali()

        content = [
            {"type": "image", "image": f"data:image/jpeg;base64,{i}"}
            for i in cl_imgs
        ]
        content.append({
            "type": "text",
            "text": (
                "Du bist ein erfahrener Dozent im Lehramtsbereich.\n"
                "Analysiere die Checkliste auf den Bildern sorgfaeltig.\n\n"
                "Die Checkliste ist eine Tabelle mit folgenden Spalten:\n"
                "- Kriterium\n"
                "- Max. Punkte\n"
                "- Erreichte Punkte\n"
                "- Anm. Erstkorrektur\n"
                "- Anmerkungen fuer Studierende\n"
                "- Anmerkung fuer Auswerter\n"
                "- Erklaerung der Bewertungskriterien\n\n"
                "Die Tabelle hat eine Oberkategorie "
                "'Motivationsfoerderliche Rueckmeldung' (8,5 Punkte gesamt). "
                "Darunter gibt es Unterkategorien (z.B. Lernziele, "
                "Individuelle Leistung, Lerndefizite, etc.) und die "
                "eigentlichen Einzelkriterien als Fragen formuliert.\n\n"
                "Extrahiere NUR die EINZELKRITERIEN (die Fragen), "
                "NICHT die Oberkategorie und NICHT die Unterkategorien.\n"
                "Punkte koennen 0.5 oder 1.0 sein.\n\n"
                "Fuer jedes Kriterium extrahiere auch den Text aus "
                "'Anmerkung fuer Auswerter' und "
                "'Erklaerung der Bewertungskriterien'.\n\n"
                "Antworte NUR im folgenden JSON-Format:\n"
                "```json\n"
                "[\n"
                "  {\n"
                '    "kriterium": "Die Frage aus der Kriterium-Spalte",\n'
                '    "max_punkte": 1.0,\n'
                '    "anmerkung_auswerter": "Text aus Anmerkung fuer Auswerter",\n'
                '    "erklaerung": "Text aus Erklaerung der Bewertungskriterien"\n'
                "  }\n"
                "]\n"
                "```\n\n"
                "Wichtig:\n"
                "- max_punkte kann 0.5 oder 1.0 sein (Dezimalzahl).\n"
                "- Extrahiere ALLE Einzelkriterien (es sollten ca. 10 sein).\n"
                "- Verwende die exakten Texte aus der Checkliste.\n"
                "- Gib NUR gueltiges JSON aus, keinen anderen Text."
            ),
        })

        raw = self._generate(content, max_tok=4096)
        criteria = parse_json_from_text(raw)

        if not criteria or not isinstance(criteria, list):
            print(f"  WARNING – raw VLM output:\n{raw}")
            return []

        print(f"\n  Extracted {len(criteria)} criteria:")
        for c in criteria:
            print(f"    - {c['kriterium'][:60]}  (max {c['max_punkte']} Pkt.)")
        return criteria

    def _phase3_search_all(
        self, criteria: List[Dict], top_k: int
    ) -> Dict[str, List]:
        """Retrieve relevant portfolio pages for every criterion."""
        print("\n" + "=" * 50)
        print("Phase 3: Retrieving Portfolio Pages per Criterion")
        print("=" * 50)
        self._free_vlm()
        self._load_colpali_index()

        results_map: Dict[str, List] = {}
        for c in criteria:
            k = c["kriterium"]
            hits = self.rag.search(k, k=top_k)
            results_map[k] = hits
            pages = ", ".join(f"p{h.page_num}" for h in hits)
            print(f"  {k[:55]:55s} -> [{pages}]")
        return results_map

    def _phase4_evaluate(
        self,
        criteria: List[Dict],
        hits_map: Dict[str, List],
        cl_imgs: List[str],
    ) -> List[Dict]:
        """Evaluate the portfolio against each criterion via VLM."""
        print("\n" + "=" * 50)
        print("Phase 4: Evaluating Portfolio per Criterion")
        print("=" * 50)
        self._free_colpali()
        self._load_vlm()

        evaluations: List[Dict] = []
        for c in criteria:
            kriterium = c["kriterium"]
            max_punkte = float(c["max_punkte"])
            anmerkung = c.get("anmerkung_auswerter", "")
            erklaerung = c.get("erklaerung", "")
            hits = hits_map.get(kriterium, [])

            print(f"\n  Criterion: {kriterium}  (max {max_punkte} Pkt.)")

            # Build multimodal content: checkliste + portfolio pages
            content = [
                {"type": "image", "image": f"data:image/jpeg;base64,{i}"}
                for i in cl_imgs
            ]
            for h in hits:
                content.append(
                    {"type": "image", "image": f"data:image/jpeg;base64,{h.base64}"}
                )
                print(f"    page {h.page_num}  score={h.score:.4f}")

            # Build evaluation prompt with extracted guidance
            prompt_parts = [
                "Du bist ein erfahrener Dozent und bewertest ein "
                "Studierenden-Portfolio.\n\n"
                "Die ersten Bilder zeigen die Bewertungs-Checkliste.\n"
                "Die weiteren Bilder zeigen relevante Seiten aus dem "
                "Studierenden-Portfolio.\n\n"
                f"Bewerte das Portfolio fuer folgendes Kriterium:\n"
                f"Kriterium: {kriterium}\n"
                f"Maximale Punkte: {max_punkte}\n\n"
            ]

            # Include extracted guidance text
            if anmerkung:
                prompt_parts.append(
                    f"Anmerkung fuer Auswerter (aus der Checkliste):\n"
                    f"{anmerkung}\n\n"
                )
            if erklaerung:
                prompt_parts.append(
                    f"Erklaerung der Bewertungskriterien (aus der Checkliste):\n"
                    f"{erklaerung}\n\n"
                )

            prompt_parts.append(
                "Anweisungen:\n"
                "1. Lies die obigen Anmerkungen und Erklaerungen genau.\n"
                "2. Pruefe im Portfolio (die Bilder nach der Checkliste), "
                "ob die beschriebenen Anforderungen erfuellt sind.\n"
                f"3. Vergib Punkte von 0 bis {max_punkte} "
                f"(in 0.5er Schritten). "
                "Bewerte FAIR und AUSGEWOGEN.\n"
                "4. Kommentar:\n"
                "   - Bei VOLLER Punktzahl: kommentar MUSS leer sein (\"\").\n"
                "   - Bei PUNKTABZUG: Du MUSST erklaeren was laut "
                "Anmerkung fuer Auswerter und Erklaerung der "
                "Bewertungskriterien fehlt oder mangelhaft ist.\n\n"
                "Antworte NUR als JSON:\n"
                "```json\n"
                '{"punkte": <Zahl>, "kommentar": "<Begruendung oder leer>"}\n'
                "```"
            )

            content.append({"type": "text", "text": "".join(prompt_parts)})

            raw = self._generate(content, max_tok=1024)
            parsed = parse_json_from_text(raw)

            if parsed and isinstance(parsed, dict):
                punkte = min(float(parsed.get("punkte", 0)), max_punkte)
                kommentar = parsed.get("kommentar", "")
                # Only keep comment when points were deducted
                if punkte >= max_punkte:
                    kommentar = ""
            else:
                punkte = 0.0
                kommentar = "Automatische Bewertung fehlgeschlagen."
                print(f"    WARNING – could not parse: {raw[:120]}")

            evaluations.append({
                "kriterium": kriterium,
                "max_punkte": max_punkte,
                "punkte": punkte,
                "kommentar": kommentar,
            })
            print(f"    => {punkte}/{max_punkte}  {kommentar[:80]}")

        return evaluations

    # ---- PDF report --------------------------------------------------------

    @staticmethod
    def _format_pts(val: float) -> str:
        """Format a point value: show as int if whole, else one decimal."""
        if val == int(val):
            return str(int(val))
        return f"{val:.1f}"

    @staticmethod
    def _make_pdf(
        evaluations: List[Dict], output_path: str, portfolio_name: str
    ):
        """Create a PDF evaluation report with a formatted table."""
        total = sum(e["punkte"] for e in evaluations)
        total_max = sum(e["max_punkte"] for e in evaluations)

        fmt = EvaluationAgent._format_pts

        pdf = FPDF(orientation="L", format="A4")  # landscape for wider table
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ---- Title ---------------------------------------------------------
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, txt="Portfolio-Bewertung", ln=True, align="C")
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 6,
            txt=f"Portfolio: {os.path.basename(portfolio_name)}",
            ln=True, align="C",
        )
        pdf.cell(
            0, 6,
            txt=f"Datum: {datetime.now().strftime('%d.%m.%Y  %H:%M')}",
            ln=True, align="C",
        )
        pdf.ln(6)

        # ---- Table ---------------------------------------------------------
        w_krit = 100
        w_punkte = 30
        w_komm = 147

        heading_style = FontFace(
            emphasis="BOLD", color=255, fill_color=(41, 65, 122)
        )
        section_style = FontFace(
            emphasis="BOLD", fill_color=(200, 210, 230)
        )
        total_style = FontFace(
            emphasis="BOLD", color=255, fill_color=(41, 65, 122)
        )

        pdf.set_font("Helvetica", "", 10)

        with pdf.table(
            col_widths=(w_krit, w_punkte, w_komm),
            headings_style=heading_style,
            line_height=pdf.font_size * 2.5,
            width=w_krit + w_punkte + w_komm,
        ) as table:
            # Header row
            hdr = table.row()
            hdr.cell("Kriterien")
            hdr.cell("Punkte")
            hdr.cell("Kommentar bei Punktabzug")

            # Section header: Motivationsfoerderliche Rueckmeldung
            row = table.row()
            row.cell(
                "Motivationsfoerderliche Rueckmeldung",
                style=section_style,
            )
            row.cell(
                f"max. {fmt(total_max)}",
                style=section_style,
                align="CENTER",
            )
            row.cell("", style=section_style)

            # Data rows (individual criteria)
            for ev in evaluations:
                row = table.row()
                row.cell(ev["kriterium"])
                row.cell(
                    f"{fmt(ev['punkte'])} / {fmt(ev['max_punkte'])}",
                    align="CENTER",
                )
                # Only show comment when there is a deduction
                if ev["punkte"] < ev["max_punkte"] and ev["kommentar"]:
                    row.cell(ev["kommentar"])
                else:
                    row.cell("-")

            # Section subtotal
            row = table.row()
            row.cell(
                "Erreichte Punkte: Motivationsfoerderliche Rueckmeldung",
                style=section_style,
            )
            row.cell(
                f"{fmt(total)} / {fmt(total_max)}",
                style=section_style,
                align="CENTER",
            )
            row.cell("", style=section_style)

            # Total row
            row = table.row()
            row.cell("Gesamte Punkte", style=total_style)
            row.cell(
                f"{fmt(total)} / {fmt(total_max)}",
                style=total_style,
                align="CENTER",
            )
            row.cell("", style=total_style)

        # ---- Save ----------------------------------------------------------
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pdf.output(output_path)
        print(f"  PDF saved: {output_path}")

    # ---- Main entry point --------------------------------------------------

    def run(
        self,
        portfolio_path: str,
        checkliste_path: str,
        output_path: str,
        top_k: int = 3,
        force_reindex: bool = False,
    ) -> List[Dict]:
        """Execute the full evaluation pipeline."""
        print("=" * 60)
        print("  Portfolio Evaluation Agent")
        print("=" * 60)
        print(f"  Portfolio  : {portfolio_path}")
        print(f"  Checkliste : {checkliste_path}")
        print(f"  Output     : {output_path}")
        print(f"  top-k      : {top_k}")
        print("=" * 60)

        # Convert checkliste pages to images (CPU only)
        print("\nConverting Checkliste to images ...")
        cl_imgs = pdf_to_base64_images(checkliste_path)
        print(f"  {len(cl_imgs)} page(s)")

        # Phase 1 – Index portfolio (ColPali on GPU)
        self._phase1_index_portfolio(portfolio_path, force_reindex)

        # Phase 2 – Extract criteria (frees ColPali, loads VLM)
        criteria = self._phase2_extract_criteria(cl_imgs)
        if not criteria:
            print("\nERROR: Could not extract criteria from Checkliste.")
            sys.exit(1)

        # Phase 3 – Retrieve pages per criterion (frees VLM, loads ColPali)
        hits_map = self._phase3_search_all(criteria, top_k)

        # Phase 4 – Evaluate each criterion (frees ColPali, loads VLM)
        evaluations = self._phase4_evaluate(criteria, hits_map, cl_imgs)

        # Phase 5 – Generate PDF report (CPU only)
        print("\n" + "=" * 50)
        print("Phase 5: Generating PDF Report")
        print("=" * 50)
        self._make_pdf(evaluations, output_path, portfolio_path)

        # Summary
        total = sum(e["punkte"] for e in evaluations)
        total_max = sum(e["max_punkte"] for e in evaluations)
        print(f"\n{'=' * 60}")
        print(f"  RESULT:  {total} / {total_max} Punkte")
        print(f"  Report:  {output_path}")
        print(f"{'=' * 60}")

        return evaluations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a student portfolio against a checklist (Visual RAG)"
    )
    parser.add_argument(
        "--portfolio", required=True,
        help="Path to the portfolio PDF",
    )
    parser.add_argument(
        "--checkliste", required=True,
        help="Path to the checklist PDF (knowledge base)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PDF path (default: results/evaluation_<timestamp>.pdf)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of portfolio pages to retrieve per criterion (default: 3)",
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Force reindex the portfolio",
    )
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/evaluation_{ts}.pdf"

    agent = EvaluationAgent()
    agent.run(
        portfolio_path=args.portfolio,
        checkliste_path=args.checkliste,
        output_path=args.output,
        top_k=args.top_k,
        force_reindex=args.reindex,
    )


if __name__ == "__main__":
    main()
