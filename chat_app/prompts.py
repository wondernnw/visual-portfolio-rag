"""German prompt templates for portfolio evaluation."""

CHECKLISTE_EXTRACTION_PROMPT = (
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
)


def build_evaluation_prompt(
    kriterium: str,
    max_punkte: float,
    anmerkung: str = "",
    erklaerung: str = "",
) -> str:
    """Build the evaluation prompt for a single criterion.

    Enhanced to request page references and quotes in the JSON output.
    """
    parts = [
        "Du bist ein erfahrener Dozent und bewertest ein "
        "Studierenden-Portfolio.\n\n"
        "Die ersten Bilder zeigen die Bewertungs-Checkliste.\n"
        "Die weiteren Bilder zeigen relevante Seiten aus dem "
        "Studierenden-Portfolio.\n\n"
        f"Bewerte das Portfolio fuer folgendes Kriterium:\n"
        f"Kriterium: {kriterium}\n"
        f"Maximale Punkte: {max_punkte}\n\n"
    ]

    if anmerkung:
        parts.append(
            f"Anmerkung fuer Auswerter (aus der Checkliste):\n"
            f"{anmerkung}\n\n"
        )
    if erklaerung:
        parts.append(
            f"Erklaerung der Bewertungskriterien (aus der Checkliste):\n"
            f"{erklaerung}\n\n"
        )

    parts.append(
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
        "Bewertungskriterien fehlt oder mangelhaft ist.\n"
        "5. Gib die Seitennummern der relevanten Portfolio-Seiten an.\n"
        "6. Zitiere kurz die relevante Textstelle aus dem Portfolio.\n\n"
        "Antworte NUR als JSON:\n"
        "```json\n"
        "{\n"
        '  "punkte": <Zahl>,\n'
        '  "kommentar": "<Begruendung oder leer>",\n'
        '  "seiten_referenzen": [<Seitennummern>],\n'
        '  "zitat": "<Kurzes Zitat aus dem Portfolio>"\n'
        "}\n"
        "```"
    )

    return "".join(parts)


SYSTEM_PROMPT_GROQ = (
    "Du bist ein KI-Assistent fuer die Bewertung von Studierenden-Portfolios "
    "im Lehramtsbereich. Du hilfst Dozenten bei der systematischen Bewertung "
    "anhand einer Checkliste.\n\n"
    "Du hast Zugriff auf folgende Werkzeuge:\n"
    "- load_bundle: Retrieval-Bundle laden (vom Cluster exportiert)\n"
    "- load_checkliste: Checkliste-Bilder laden (fuer Bewertungskontext)\n"
    "- evaluate_criterion: Ein einzelnes Kriterium bewerten\n"
    "- evaluate_all: Alle Kriterien auf einmal bewerten\n"
    "- generate_report: PDF-Bericht erstellen\n\n"
    "Typischer Ablauf:\n"
    "1. Retrieval-Bundle laden\n"
    "2. Optional: Checkliste laden\n"
    "3. Kriterien bewerten (einzeln oder alle)\n"
    "4. PDF-Bericht generieren\n\n"
    "Antworte auf Deutsch, sachlich und strukturiert."
)
