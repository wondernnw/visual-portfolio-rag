"""PDF report generation for portfolio evaluations.

Extracted from evaluate_portfolio.py with added page reference display.
"""

import os
from datetime import datetime
from typing import Dict, List

from fpdf import FPDF

try:
    from fpdf import FontFace
except ImportError:
    from fpdf.fonts import FontFace


def format_pts(val: float) -> str:
    """Format a point value: show as int if whole, else one decimal."""
    if val == int(val):
        return str(int(val))
    return f"{val:.1f}"


def make_pdf(
    evaluations: List[Dict],
    output_path: str,
    portfolio_name: str,
):
    """Create a PDF evaluation report with a formatted table.

    Enhanced to include page references (Seitenverweis) column.
    """
    total = sum(e["punkte"] for e in evaluations)
    total_max = sum(e["max_punkte"] for e in evaluations)

    pdf = FPDF(orientation="L", format="A4")  # landscape
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Title -------------------------------------------------------------
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

    # ---- Table -------------------------------------------------------------
    w_krit = 85
    w_punkte = 25
    w_komm = 110
    w_seite = 57

    heading_style = FontFace(
        emphasis="BOLD", color=255, fill_color=(41, 65, 122)
    )
    section_style = FontFace(
        emphasis="BOLD", fill_color=(200, 210, 230)
    )
    total_style = FontFace(
        emphasis="BOLD", color=255, fill_color=(41, 65, 122)
    )

    pdf.set_font("Helvetica", "", 9)

    with pdf.table(
        col_widths=(w_krit, w_punkte, w_komm, w_seite),
        headings_style=heading_style,
        line_height=pdf.font_size * 2.5,
        width=w_krit + w_punkte + w_komm + w_seite,
    ) as table:
        # Header row
        hdr = table.row()
        hdr.cell("Kriterien")
        hdr.cell("Punkte")
        hdr.cell("Kommentar bei Punktabzug")
        hdr.cell("Seitenverweis / Zitat")

        # Section header
        row = table.row()
        row.cell(
            "Motivationsfoerderliche Rueckmeldung",
            style=section_style,
        )
        row.cell(
            f"max. {format_pts(total_max)}",
            style=section_style,
            align="CENTER",
        )
        row.cell("", style=section_style)
        row.cell("", style=section_style)

        # Data rows
        for ev in evaluations:
            row = table.row()
            row.cell(ev["kriterium"])
            row.cell(
                f"{format_pts(ev['punkte'])} / {format_pts(ev['max_punkte'])}",
                align="CENTER",
            )

            # Comment
            if ev["punkte"] < ev["max_punkte"] and ev.get("kommentar"):
                row.cell(ev["kommentar"])
            else:
                row.cell("-")

            # Page refs + quote
            refs = ev.get("seiten_referenzen", [])
            zitat = ev.get("zitat", "")
            ref_parts = []
            if refs:
                ref_parts.append("S. " + ", ".join(str(r) for r in refs))
            if zitat:
                short = zitat[:80] + "..." if len(zitat) > 80 else zitat
                ref_parts.append(f'"{short}"')
            row.cell(" | ".join(ref_parts) if ref_parts else "-")

        # Section subtotal
        row = table.row()
        row.cell(
            "Erreichte Punkte: Motivationsfoerderliche Rueckmeldung",
            style=section_style,
        )
        row.cell(
            f"{format_pts(total)} / {format_pts(total_max)}",
            style=section_style,
            align="CENTER",
        )
        row.cell("", style=section_style)
        row.cell("", style=section_style)

        # Total row
        row = table.row()
        row.cell("Gesamte Punkte", style=total_style)
        row.cell(
            f"{format_pts(total)} / {format_pts(total_max)}",
            style=total_style,
            align="CENTER",
        )
        row.cell("", style=total_style)
        row.cell("", style=total_style)

    # ---- Save --------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pdf.output(output_path)
    print(f"  PDF saved: {output_path}")
