"""
pdf_report.py
=============
Génère un rapport PDF synthétique pour FinHealth Analyzer.
Utilise fpdf2 pour la génération.
"""

from __future__ import annotations

import io
import textwrap
from datetime import datetime
from typing import Optional

import pandas as pd
from fpdf import FPDF


# ─────────────────────────────────────────────
# Couleurs du thème premium
# ─────────────────────────────────────────────
_BG = (2, 6, 23)
_SURFACE = (15, 23, 42)
_ACCENT = (56, 189, 248)
_TEXT = (248, 250, 252)
_MUTED = (148, 163, 184)
_GREEN = (16, 185, 129)
_AMBER = (245, 158, 11)
_RED = (239, 68, 68)


class FinHealthPDF(FPDF):
    """PDF premium dark-mode pour les rapports financiers."""

    def __init__(self, company_name: str, ticker: str, **kwargs):
        super().__init__(**kwargs)
        self.company_name = self._sanitize(company_name)
        self.ticker = ticker
        self.set_auto_page_break(auto=True, margin=20)

    @staticmethod
    def _sanitize(text: str) -> str:
        """Replace non-Latin-1 chars with safe ASCII equivalents."""
        replacements = {
            "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
            "\u00b7": ".", "\u2212": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Final fallback: encode to latin-1 and drop anything remaining
        return text.encode("latin-1", errors="replace").decode("latin-1")

    # ── Header / Footer ──

    def header(self):
        self.set_fill_color(*_BG)
        self.rect(0, 0, 210, 297, "F")  # Full page BG
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_ACCENT)
        self.cell(0, 8, self._sanitize("FINHEALTH ANALYZER  -  RAPPORT PRO"), align="L")
        self.set_text_color(*_MUTED)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 8, datetime.now().strftime("%d/%m/%Y %H:%M"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_ACCENT)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  FinHealth Analyzer v2.1  |  {self.company_name}", align="C")

    # ── Helpers ──

    def _section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_ACCENT)
        self.cell(0, 8, self._sanitize(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(56, 189, 248)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def _kv(self, label: str, value: str, color: tuple = _TEXT):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_MUTED)
        self.cell(60, 6, self._sanitize(label), new_x="RIGHT")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 6, self._sanitize(value), new_x="LMARGIN", new_y="NEXT")

    def _multiline(self, text: str, width: int = 190):
        """Write a block of text with line wrapping."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_TEXT)
        for line in text.split("\n"):
            wrapped = textwrap.wrap(line, width=95) or [""]
            for wl in wrapped:
                self.cell(0, 5, self._sanitize(wl), new_x="LMARGIN", new_y="NEXT")

    def _score_color(self, val: Optional[float], low: float, high: float) -> tuple:
        if val is None:
            return _MUTED
        if val >= high:
            return _GREEN
        if val <= low:
            return _RED
        return _AMBER


def generate_pdf_report(
    company_info: dict,
    latest: pd.Series,
    metrics_df: pd.DataFrame,
    ai_diagnostic: str = "",
) -> bytes:
    """
    Génère un rapport PDF complet et retourne les bytes du document.

    Paramètres :
    - company_info : dict avec name, sector, industry, country
    - latest : pd.Series de la dernière année (avec métriques calculées)
    - metrics_df : DataFrame complet des métriques
    - ai_diagnostic : texte du dernier diagnostic IA (optionnel)
    """
    from data_handler import format_large_number, format_percent, format_ratio

    name = company_info.get("name", "Entreprise")
    ticker = company_info.get("ticker", "")

    pdf = FinHealthPDF(company_name=name, ticker=ticker)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Titre principal ──
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*_TEXT)
    pdf.cell(0, 12, pdf._sanitize(name), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_MUTED)
    sector = company_info.get("sector", "N/A")
    industry = company_info.get("industry", "N/A")
    country = company_info.get("country", "N/A")
    pdf.cell(0, 6, pdf._sanitize(f"{sector}  |  {industry}  |  {country}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── KPIs ──
    pdf._section_title("Indicateurs Clés de Performance")

    def _safe(col: str) -> Optional[float]:
        v = latest.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)

    kpis = [
        ("Chiffre d'Affaires", format_large_number(_safe("revenue")), _TEXT),
        ("Résultat Net", format_large_number(_safe("net_income")), _GREEN if (_safe("net_income") or 0) > 0 else _RED),
        ("Total Actif", format_large_number(_safe("total_assets")), _TEXT),
        ("Capitaux Propres", format_large_number(_safe("shareholders_equity")), _TEXT),
        ("Trésorerie", format_large_number(_safe("cash")), _TEXT),
    ]
    for label, val, color in kpis:
        pdf._kv(label, val, color)

    pdf.ln(2)

    # ── Ratios de Rentabilité ──
    pdf._section_title("Rentabilité")
    ratios_rent = [
        ("Marge Nette", format_percent(_safe("net_margin")), pdf._score_color(_safe("net_margin"), 0.05, 0.15)),
        ("Marge Brute", format_percent(_safe("gross_margin")), pdf._score_color(_safe("gross_margin"), 0.15, 0.40)),
        ("ROE", format_percent(_safe("roe")), pdf._score_color(_safe("roe"), 0.05, 0.15)),
        ("ROA", format_percent(_safe("roa")), pdf._score_color(_safe("roa"), 0.02, 0.08)),
    ]
    for label, val, color in ratios_rent:
        pdf._kv(label, val, color)

    # ── Ratios de Liquidité ──
    pdf._section_title("Liquidité")
    ratios_liq = [
        ("Current Ratio", format_ratio(_safe("current_ratio")), pdf._score_color(_safe("current_ratio"), 1.0, 2.0)),
        ("Quick Ratio", format_ratio(_safe("quick_ratio")), pdf._score_color(_safe("quick_ratio"), 0.8, 1.5)),
        ("Fonds de Roulement", format_large_number(_safe("working_capital")), _TEXT),
    ]
    for label, val, color in ratios_liq:
        pdf._kv(label, val, color)

    # ── Solvabilité ──
    pdf._section_title("Solvabilité & Levier")
    de = _safe("d_e_ratio")
    ic = _safe("interest_coverage")
    ratios_solv = [
        ("Debt / Equity", format_ratio(de), pdf._score_color(de, 0.5, 2.0) if de and de <= 2 else _RED if de else _MUTED),
        ("Debt / Assets", format_ratio(_safe("d_a_ratio")), _TEXT),
        ("Couverture Intérêts", format_ratio(ic), pdf._score_color(ic, 2.0, 5.0)),
    ]
    for label, val, color in ratios_solv:
        pdf._kv(label, val, color)

    # ── Scores ──
    pdf._section_title("Scores Avancés")
    z = _safe("z_score")
    f = _safe("f_score")
    hs = _safe("health_score")

    z_label = "Sûr" if z and z > 2.99 else ("Gris" if z and z > 1.81 else "Danger") if z else "N/A"
    f_label = f"{int(f)} / 9" if f else "N/A"
    hs_label = f"{hs:.1f} / 100" if hs else "N/A"

    pdf._kv("Altman Z-Score", f"{z:.2f} ({z_label})" if z else "N/A", pdf._score_color(z, 1.81, 2.99))
    pdf._kv("Piotroski F-Score", f_label, pdf._score_color(f, 4, 8))
    pdf._kv("Score Santé Composite", hs_label, pdf._score_color(hs, 40, 65))

    # ── Diagnostic IA ──
    if ai_diagnostic:
        pdf._section_title("Diagnostic IA (Executive Summary)")
        # Clean markdown bold markers for PDF
        clean_diag = ai_diagnostic.replace("**", "").replace("*", "")
        pdf._multiline(clean_diag)

    # ── Piotroski Radar data (textual) ──
    f_details = latest.get("f_score_details")
    if f_details and isinstance(f_details, dict):
        details = f_details.get("details", {})
        if details:
            pdf._section_title("Détail Piotroski F-Score")
            for crit, passed in details.items():
                icon = "PASS" if passed == 1 else "FAIL"
                color = _GREEN if passed == 1 else _RED
                pdf._kv(crit, icon, color)

    # ── Finalize ──
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
