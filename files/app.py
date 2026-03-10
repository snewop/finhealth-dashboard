"""
app.py
======
Interface Streamlit — orchestre data_handler.py et finance_metrics.py.
Lancer avec : streamlit run app.py
"""

from __future__ import annotations

import traceback
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_handler import (
    apply_mapping,
    clean_financial_df,
    compute_working_capital,
    detect_column_mapping,
    format_large_number,
    format_percent,
    format_ratio,
    load_from_file,
    load_from_yfinance,
    safe_get,
)
from finance_metrics import (
    altman_z_score,
    altman_z_score_label,
    cash_ratio,
    composite_health_score,
    current_ratio,
    debt_to_assets,
    debt_to_equity,
    ebitda_margin,
    gross_profit_margin,
    interest_coverage,
    net_profit_margin,
    piotroski_f_score,
    piotroski_label,
    quick_ratio,
    return_on_assets,
    return_on_equity,
)

# ─────────────────────────────────────────────
# Configuration globale
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FinHealth Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design CSS personnalisé ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    :root {
        --bg: #0a0f1a;
        --surface: #111827;
        --surface2: #1a2235;
        --border: #1e2d45;
        --accent: #3b82f6;
        --accent2: #06b6d4;
        --text: #e2e8f0;
        --muted: #64748b;
        --green: #22c55e;
        --amber: #f59e0b;
        --red: #ef4444;
    }

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

    .stApp { background-color: var(--bg) !important; }

    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    .metric-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        position: relative; /* Pour le tooltip absolu */
    }
    .metric-card:hover { 
        border-color: var(--accent); 
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    }

    /* Info-bulles pur CSS */
    .tooltip-container {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--muted);
        cursor: help;
    }
    
    .tooltip-container .tooltip-text {
        visibility: hidden;
        width: 220px;
        background-color: var(--surface);
        color: var(--text);
        text-transform: none;
        letter-spacing: normal;
        text-align: center;
        border-radius: 8px;
        padding: 8px;
        font-size: 11px;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        
        /* Position */
        position: absolute;
        z-index: 50;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        
        /* Animation */
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip-container .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: var(--surface) transparent transparent transparent;
    }

    .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    .metric-label {
        font-size: 11px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
        color: var(--text);
        font-family: 'Syne', sans-serif;
    }
    .metric-delta {
        font-size: 11px;
        margin-top: 3px;
    }

    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--accent);
        margin: 24px 0 12px 0;
        border-left: 3px solid var(--accent);
        padding-left: 10px;
    }

    .info-chip {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 11px;
        display: inline-block;
        margin: 2px;
        color: var(--muted);
    }

    .stButton>button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Mono', monospace !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
        transition: opacity 0.2s !important;
    }
    .stButton>button:hover { opacity: 0.85 !important; }

    div[data-testid="metric-container"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 14px !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 13px !important;
    }

    .stDataFrame { border-radius: 8px !important; }

    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid var(--amber);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: var(--amber);
        margin: 8px 0;
    }
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid var(--green);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: var(--green);
        margin: 8px 0;
    }
    .danger-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid var(--red);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: var(--red);
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = {
    "paper_bgcolor": "#0a0f1a",
    "plot_bgcolor": "#111827",
    "font_color": "#94a3b8",
    "gridcolor": "#1e2d45",
    "colorscale": px.colors.sequential.Blues_r,
}


def plotly_layout(fig: go.Figure, title: str = "", height: int = 320) -> go.Figure:
    """Applique le thème sombre cohérent à une figure Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=14, color="#e2e8f0")),
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(color=PLOTLY_THEME["font_color"], family="DM Mono, monospace"),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=PLOTLY_THEME["gridcolor"], showgrid=False),
        yaxis=dict(gridcolor=PLOTLY_THEME["gridcolor"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─────────────────────────────────────────────
# Calculs par ligne du DataFrame
# ─────────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame, market_data: dict = {}) -> pd.DataFrame:
    """
    Calcule tous les ratios et scores pour chaque ligne (année) du DataFrame.
    Les nouvelles colonnes sont ajoutées directement.
    """
    result_rows = []

    for i, (_, row) in enumerate(df.iterrows()):
        r = row.to_dict()

        year = int(r.get("year", 0))
        mkt = market_data.get(year, {})

        rev = safe_get(row, "revenue")
        ni = safe_get(row, "net_income")
        gp = safe_get(row, "gross_profit")
        eb = safe_get(row, "ebitda")
        ebit = safe_get(row, "ebit")
        ta = safe_get(row, "total_assets")
        ca = safe_get(row, "current_assets")
        inv = safe_get(row, "inventories")
        cash_val = safe_get(row, "cash")
        cl = safe_get(row, "current_liabilities")
        tl = safe_get(row, "total_liabilities")
        td = safe_get(row, "total_debt")
        eq = safe_get(row, "shareholders_equity")
        re_ = safe_get(row, "retained_earnings")
        ocf = safe_get(row, "operating_cash_flow")
        ie = safe_get(row, "interest_expense")
        shares = safe_get(row, "shares_outstanding") or mkt.get("shares_outstanding")
        mkt_cap = mkt.get("market_cap")

        wc = compute_working_capital(row)

        # Ratios
        r["net_margin"] = net_profit_margin(ni, rev)
        r["gross_margin"] = gross_profit_margin(gp, rev)
        r["ebitda_margin"] = ebitda_margin(eb, rev)
        r["roe"] = return_on_equity(ni, eq)
        r["roa"] = return_on_assets(ni, ta)
        r["current_ratio"] = current_ratio(ca, cl)
        r["quick_ratio"] = quick_ratio(ca, inv or 0.0, cl)
        r["cash_ratio"] = cash_ratio(cash_val, cl)
        r["d_e_ratio"] = debt_to_equity(td, eq)
        r["d_a_ratio"] = debt_to_assets(td, ta)
        r["interest_coverage"] = interest_coverage(ebit, ie)
        r["working_capital"] = wc

        # Altman Z-Score
        r["z_score"] = altman_z_score(
            working_capital=wc or 0.0,
            retained_earnings=re_ or 0.0,
            ebit=ebit or 0.0,
            market_cap=mkt_cap or 0.0,
            total_liabilities=tl or 0.0,
            revenue=rev or 0.0,
            total_assets=ta,
        )

        result_rows.append(r)

    metrics_df = pd.DataFrame(result_rows)

    # Piotroski F-Score (nécessite la ligne précédente)
    f_scores = []
    for i in range(len(metrics_df)):
        row = metrics_df.iloc[i]
        prev = metrics_df.iloc[i - 1] if i > 0 else None

        def mg(col: str, src=row) -> Optional[float]:
            v = src.get(col)
            return float(v) if v is not None and not pd.isna(v) else None

        def mprev(col: str) -> Optional[float]:
            if prev is None:
                return None
            return mg(col, prev)

        lev_c = mg("d_a_ratio")
        lev_p = mprev("d_a_ratio")

        f_dict = piotroski_f_score(
            net_income=mg("net_income") or 0.0,
            operating_cash_flow=mg("operating_cash_flow") or 0.0,
            roa_current=mg("roa") or 0.0,
            roa_previous=mprev("roa") or 0.0,
            accruals=0.0,
            leverage_current=lev_c or 0.0,
            leverage_previous=lev_p or 0.0,
            current_ratio_current=mg("current_ratio") or 0.0,
            current_ratio_previous=mprev("current_ratio") or 0.0,
            shares_current=mg("shares_outstanding") or 1.0,
            shares_previous=mprev("shares_outstanding") or 1.0,
            gross_margin_current=mg("gross_margin") or 0.0,
            gross_margin_previous=mprev("gross_margin") or 0.0,
            asset_turnover_current=(mg("revenue") or 0.0) / (mg("total_assets") or 1.0),
            asset_turnover_previous=(mprev("revenue") or 0.0) / (mprev("total_assets") or 1.0),
            total_assets=mg("total_assets") or 1.0,
        )
        f_scores.append(f_dict)

    metrics_df["f_score"] = [d["total"] for d in f_scores]
    metrics_df["f_score_details"] = f_scores

    # Composite score
    metrics_df["health_score"] = metrics_df.apply(
        lambda r: composite_health_score(
            net_margin=r.get("net_margin"),
            roe=r.get("roe"),
            current_ratio_val=r.get("current_ratio"),
            debt_equity=r.get("d_e_ratio"),
            z_score=r.get("z_score"),
            f_score=r.get("f_score"),
        ),
        axis=1,
    )

    return metrics_df


# ─────────────────────────────────────────────
# Composants UI réutilisables
# ─────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = "", color: str = "#e2e8f0", tooltip: str = "") -> None:
    delta_html = f'<div class="metric-delta" style="color:{color}">{delta}</div>' if delta else ""
    
    # Intégration du tooltip si fourni
    label_html = f"""
    <div class="tooltip-container">
        {label}
        <span class="tooltip-text">{tooltip}</span>
    </div>
    """ if tooltip else f"{label}"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label_html}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def color_for_value(val: Optional[float], low: float, high: float, inverse: bool = False) -> str:
    """Retourne une couleur selon que la valeur est dans la zone verte, ambre ou rouge."""
    if val is None:
        return "#94a3b8"
    good = val >= high if not inverse else val <= low
    bad = val <= low if not inverse else val >= high
    if good:
        return "#22c55e"
    elif bad:
        return "#ef4444"
    return "#f59e0b"


# ─────────────────────────────────────────────
# Sections du dashboard
# ─────────────────────────────────────────────

def render_kpi_header(latest: pd.Series, currency: str = "USD") -> None:
    """Affiche les KPI principaux en haut du dashboard avec Plotly haut de gamme."""
    section_header("📊 Indicateurs Clés — Dernière Année")
    cols = st.columns(5)

    kpis = [
        ("Chiffre d'Affaires", latest.get("revenue"), "revenue"),
        ("Résultat Net", latest.get("net_income"), "net_income"),
        ("Total Actif", latest.get("total_assets"), "total_assets"),
        ("Capitaux Propres", latest.get("shareholders_equity"), "shareholders_equity"),
        ("Trésorerie", latest.get("cash"), "cash"),
    ]

    for col, (label, value, _) in zip(cols, kpis):
        with col:
            val_formatted = format_large_number(value, currency)
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number",
                value=value if value and not pd.isna(value) else 0,
                number={'valueformat': '.2s', 'prefix': '$' if currency == 'USD' else '€'},
                title={"text": label, "font": {"size": 14, "color": "#94a3b8", "family": "Syne"}},
                domain={'x': [0, 1], 'y': [0, 1]},
            ))
            fig.update_layout(
                paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
                plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
                font=dict(color="#e2e8f0", family="DM Mono, monospace"),
                height=120,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            # Custom formatting if Plotly formatting is not ideal
            fig.data[0].number.valueformat = ""
            fig.data[0].value = 0 # Avoid raw big numbers display issues
            fig.update_traces(number={'font': {'size': 26, 'color': '#3b82f6', 'family': 'Syne'}})
            
            # Use raw markdown if plotly numbers are too constrained for complex currency formatting
            # but user requested Plotly indicators. We'll use a styled Plotly indicator visually.
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Correction: The user asked for Plotly indicators, but formatting strings for diverse currencies is tricky in pure Plotly Indicators.
            # Let's use the layout trick to show custom text while keeping the Plotly aesthetic.
            fig.layout.annotations = [{
                'text': val_formatted,
                'x': 0.5, 'y': 0.45,
                'showarrow': False,
                'font': {'size': 28, 'color': '#3b82f6', 'family': 'Syne', 'weight': 'bold'}
            }]
            fig.update_traces(number={'font': {'size': 1}}) # hide original number
            # We re-inject to display correctly.
            col.empty() # Clear previous
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_executive_summary(latest: pd.Series, prev: Optional[pd.Series]) -> None:
    """Génère un diagnostic textuel basé sur les résultats de la dernière année (Premium Feature)."""
    section_header("🤖 Executive Summary")
    
    sentences = []
    
    # Check CA
    rev_c = latest.get("revenue")
    rev_p = prev.get("revenue") if prev is not None else None
    if rev_c and rev_p and rev_p > 0:
        croissance = (rev_c - rev_p) / rev_p
        if croissance > 0.1:
            sentences.append(f"🟢 **Forte croissance** des revenus retenue à **+{format_percent(croissance)}**, consolidant la dynamique commerciale.")
        elif croissance > 0:
            sentences.append(f"🟢 **Croissance modérée** des revenus à **+{format_percent(croissance)}**.")
        else:
            sentences.append(f"🔴 **Contraction du chiffre d'affaires** observée de **{format_percent(croissance)}** d'une année sur l'autre.")

    # Check Rentabilité Nette
    nm = latest.get("net_margin")
    if nm is not None:
        if nm > 0.1:
            sentences.append(f"🟢 Excellente **rentabilité nette** ({format_percent(nm)}), indiquant un très bon contrôle des coûts et un pricing power sain.")
        elif nm > 0:
            sentences.append(f"🟡 **Rentabilité positive** mais contenue ({format_percent(nm)}), des marges de manœuvre d'optimisation existent.")
        else:
            sentences.append(f"🔴 La société est **en perte** avec une marge nette de {format_percent(nm)}.")
            
    # Check Liquidité
    cr = latest.get("current_ratio")
    if cr is not None:
        if cr < 1:
            sentences.append(f"🔴 **Alerte Liquidité** : Le ratio de liquidité générale ({cr:.2f}x) est sous 1, signifiant un risque de tension financière à court terme.")
        elif cr > 2:
            sentences.append(f"🟡 **Sur-liquidité possible** : Le ratio de liquidité générale est très élevé ({cr:.2f}x), suggérant une allocation de capital peut-être sous-optimale.")
        else:
            sentences.append(f"🟢 **Liquidité saine** ({cr:.2f}x), l'entreprise couvre confortablement ses obligations de court terme.")

    # Check Score Sante
    hs = latest.get("health_score")
    if hs is not None:
        if hs >= 70:
            sentences.append(f"🏆 Le diagnostic de santé global est **Excellent** (Score: {hs:.1f}/100).")
        elif hs >= 40:
            sentences.append(f"⚖️ Le diagnostic de santé global est **Moyen/Stable** (Score: {hs:.1f}/100).")
        else:
            sentences.append(f"⚠️ Le diagnostic de santé global est **Préoccupant** (Score: {hs:.1f}/100).")

    if not sentences:
        summary_text = "Pas assez de données pour générer un diagnostic pertinent."
    else:
        summary_text = " ".join(sentences)

    st.markdown(f"""
    <div style="background:var(--surface2);border-left:4px solid var(--accent);border-radius:0 8px 8px 0;padding:16px 20px;margin-bottom:20px;box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <p style="font-size:14px;line-height:1.6;color:var(--text);margin:0;">
            {summary_text}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_profitability(latest: pd.Series, prev: Optional[pd.Series]) -> None:
    """Section rentabilité avec tooltips."""
    section_header("💹 Rentabilité")
    cols = st.columns(4)
    metrics = [
        ("Marge Nette", latest.get("net_margin"), prev.get("net_margin") if prev is not None else None,
         format_percent, 0.05, 0.20, False, "Résultat net divisé par le chiffre d'affaires. Indique la proportion de bénéfice final."),
        ("Marge Brute", latest.get("gross_margin"), prev.get("gross_margin") if prev is not None else None,
         format_percent, 0.15, 0.40, False, "Différence entre le chiffre d'affaires et le coût des ventes, en % du CA."),
        ("ROE", latest.get("roe"), prev.get("roe") if prev is not None else None,
         format_percent, 0.05, 0.15, False, "Return on Equity (Rentabilité des capitaux propres) : Efficience avec laquelle l'entreprise génère du profit grâce à ses fonds propres."),
        ("ROA", latest.get("roa"), prev.get("roa") if prev is not None else None,
         format_percent, 0.02, 0.08, False, "Return on Assets (Rentabilité économique) : Mesure de l'efficacité d'utilisation des actifs pour générer du bénéfice."),
    ]
    for col, (label, val, prev_val, fmt, low, high, inv, tooltip) in zip(cols, metrics):
        with col:
            color = color_for_value(val, low, high, inv)
            delta = ""
            if prev_val is not None and val is not None:
                diff = val - prev_val
                sign = "▲" if diff >= 0 else "▼"
                delta = f"{sign} {fmt(abs(diff))}"
            metric_card(label, fmt(val), delta, color, tooltip)


def render_liquidity(latest: pd.Series, prev: Optional[pd.Series]) -> None:
    """Section liquidité avec tooltips."""
    section_header("💧 Liquidité")
    cols = st.columns(4)
    metrics = [
        ("Current Ratio", latest.get("current_ratio"), prev.get("current_ratio") if prev is not None else None,
         format_ratio, 1.0, 2.0, False, "Ratio de liquidité générale : Actifs courants / Passifs courants. > 1 est sain."),
        ("Quick Ratio", latest.get("quick_ratio"), prev.get("quick_ratio") if prev is not None else None,
         format_ratio, 0.8, 1.5, False, "Ratio de liquidité immédiate : Exclut les stocks. Plus strict que le Current Ratio."),
        ("Cash Ratio", latest.get("cash_ratio"), prev.get("cash_ratio") if prev is not None else None,
         format_ratio, 0.2, 0.5, False, "Trésorerie / Passifs courants. Capacité ultra-immédiate à payer les dettes."),
        ("Fonds de Roulement", None, None, lambda v: format_large_number(v), 0, 1, False, "Capital Working: Actif courant - Passif courant."),
    ]
    # Override fonds de roulement
    metrics[3] = ("Fonds de Roulement", latest.get("working_capital"),
                  prev.get("working_capital") if prev is not None else None,
                  lambda v: format_large_number(v), 0, 1e9, False, "Capital Working: Actif courant - Passif courant.")

    for col, (label, val, prev_val, fmt, low, high, inv, tooltip) in zip(cols, metrics):
        with col:
            color = color_for_value(val, low, high, inv)
            delta = ""
            if prev_val is not None and val is not None:
                try:
                    diff = val - prev_val
                    sign = "▲" if diff >= 0 else "▼"
                    delta = f"{sign} {fmt(abs(diff))}"
                except Exception:
                    pass
            metric_card(label, fmt(val), delta, color, tooltip)


def render_solvency(latest: pd.Series, prev: Optional[pd.Series]) -> None:
    """Section solvabilité / levier avec tooltips."""
    section_header("🏦 Solvabilité & Levier")
    cols = st.columns(3)
    metrics = [
        ("Debt / Equity", latest.get("d_e_ratio"), prev.get("d_e_ratio") if prev is not None else None,
         format_ratio, 0.5, 2.0, True, "Dette Totale / Capitaux Propres. Mesure l'exposition au levier. >2 est souvent risqué."),
        ("Debt / Assets", latest.get("d_a_ratio"), prev.get("d_a_ratio") if prev is not None else None,
         format_ratio, 0.3, 0.6, True, "Plus il est élevé, plus l'entreprise est risquée et financée par la dette."),
        ("Couv. Intérêts", latest.get("interest_coverage"), prev.get("interest_coverage") if prev is not None else None,
         format_ratio, 2.0, 5.0, False, "Capacité à payer les charges d'intérêts grâce à l'EBIT. <1.5 est un Warning majeur."),
    ]
    for col, (label, val, prev_val, fmt, low, high, inv, tooltip) in zip(cols, metrics):
        with col:
            color = color_for_value(val, low, high, inv)
            delta = ""
            if prev_val is not None and val is not None:
                try:
                    diff = val - prev_val
                    sign = "▲" if diff >= 0 else "▼"
                    delta = f"{sign} {fmt(abs(diff))}"
                except Exception:
                    pass
            metric_card(label, fmt(val), delta, color, tooltip)


def render_scores(latest: pd.Series) -> None:
    """Section scores avancés (Altman, Piotroski, Composite)."""
    section_header("🔬 Scores Avancés")

    col1, col2, col3 = st.columns(3)

    # ── Altman Z-Score ──
    with col1:
        z = latest.get("z_score")
        z_label, z_color = altman_z_score_label(z)
        z_val = f"{z:.2f}" if z is not None else "N/A"

        st.markdown(f"""
        <div class="metric-card" style="border-color:{z_color}40">
            <div class="metric-label">Altman Z-Score</div>
            <div class="metric-value" style="font-size:28px;color:{z_color}">{z_val}</div>
            <div style="margin-top:6px"><span class="score-badge" style="background:{z_color}20;color:{z_color}">{z_label}</span></div>
            <div style="margin-top:8px;font-size:11px;color:#64748b">
                &gt;2.99 Sûr · 1.81–2.99 Gris · &lt;1.81 Danger
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Piotroski F-Score ──
    with col2:
        f = latest.get("f_score")
        f_int = int(f) if f is not None and not pd.isna(f) else None
        f_label, f_color = piotroski_label(f_int or 0)
        f_val = str(f_int) + " / 9" if f_int is not None else "N/A"

        st.markdown(f"""
        <div class="metric-card" style="border-color:{f_color}40">
            <div class="metric-label">Piotroski F-Score</div>
            <div class="metric-value" style="font-size:28px;color:{f_color}">{f_val}</div>
            <div style="margin-top:6px"><span class="score-badge" style="background:{f_color}20;color:{f_color}">{f_label}</span></div>
            <div style="margin-top:8px;font-size:11px;color:#64748b">
                8-9 Solide · 4-7 Neutre · 0-3 Faible
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Score Composite ──
    with col3:
        hs = latest.get("health_score")
        hs_val = f"{hs:.1f} / 100" if hs is not None else "N/A"
        hs_color = "#22c55e" if (hs or 0) >= 65 else ("#f59e0b" if (hs or 0) >= 40 else "#ef4444")

        # Gauge Plotly
        if hs is not None:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hs,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Score Santé", "font": {"color": "#94a3b8", "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#334155"},
                    "bar": {"color": hs_color},
                    "bgcolor": "#1a2235",
                    "steps": [
                        {"range": [0, 40], "color": "#1f1018"},
                        {"range": [40, 65], "color": "#1f1a0e"},
                        {"range": [65, 100], "color": "#0e1f12"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": hs,
                    },
                },
                number={"font": {"color": hs_color, "family": "Syne"}},
            ))
            plotly_layout(fig, height=200)
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Score Santé Composite</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)


def render_piotroski_details(latest: pd.Series) -> None:
    """Affiche le détail des 9 critères du F-Score sous forme de Plotly Polar radar chart Dark Mode."""
    f_details = latest.get("f_score_details")
    if not f_details or not isinstance(f_details, dict):
        return

    section_header("📋 Détail Piotroski F-Score (Radar)")
    details = f_details.get("details", {})
    
    # Plotly radar
    categories = list(details.keys())
    values = list(details.values())

    # Close the radar loop
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=2),
        marker=dict(
            color=['#22c55e' if v == 1 else '#ef4444' for v in values],
            size=6
        ),
        name='Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 1],
                ticktext=["Échec", "Validation"],
                gridcolor='#1e2d45',
                color='#94a3b8'
            ),
            angularaxis=dict(
                gridcolor='#1e2d45',
                linecolor='#1e2d45',
                tickfont=dict(color="#e2e8f0", size=11, family="DM Mono")
            ),
            bgcolor='#111827'
        ),
        showlegend=False,
        paper_bgcolor='#0a0f1a',
        plot_bgcolor='#111827',
        font=dict(color='#e2e8f0', family='DM Mono'),
        height=380,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_charts(metrics_df: pd.DataFrame, currency: str = "USD") -> None:
    """Graphiques interactifs sur l'évolution temporelle."""
    section_header("📈 Évolution Temporelle")

    df = metrics_df.sort_values("year")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Revenus & Marges", "📐 Liquidité & Levier",
        "🏆 Scores", "📊 Rentabilité"
    ])

    # ── Tab 1 : Revenus & Marges ──
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if "revenue" in df.columns and df["revenue"].notna().any():
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df["year"], y=df["revenue"] / 1e9,
                    name="CA", marker_color="#3b82f6",
                ))
                if "net_income" in df.columns:
                    fig.add_trace(go.Bar(
                        x=df["year"], y=df["net_income"] / 1e9,
                        name="Résultat Net", marker_color="#22c55e",
                    ))
                plotly_layout(fig, "CA vs Résultat Net (Mrd)")
                fig.update_layout(barmode="group")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            margin_cols = {
                "net_margin": ("Marge Nette", "#3b82f6"),
                "gross_margin": ("Marge Brute", "#22c55e"),
                "ebitda_margin": ("Marge EBITDA", "#f59e0b"),
            }
            fig = go.Figure()
            for col_name, (label, color) in margin_cols.items():
                if col_name in df.columns and df[col_name].notna().any():
                    fig.add_trace(go.Scatter(
                        x=df["year"], y=df[col_name] * 100,
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                    ))
            plotly_layout(fig, "Évolution des Marges (%)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2 : Liquidité & Levier ──
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for col_name, (label, color) in [
                ("current_ratio", ("Current Ratio", "#3b82f6")),
                ("quick_ratio", ("Quick Ratio", "#06b6d4")),
            ]:
                if col_name in df.columns and df[col_name].notna().any():
                    fig.add_trace(go.Scatter(
                        x=df["year"], y=df[col_name],
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                    ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444",
                          annotation_text="Seuil critique (1x)")
            plotly_layout(fig, "Ratios de Liquidité")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "d_e_ratio" in df.columns and df["d_e_ratio"].notna().any():
                fig = go.Figure(go.Bar(
                    x=df["year"], y=df["d_e_ratio"],
                    marker_color=["#ef4444" if v > 2.0 else "#f59e0b" if v > 1.0 else "#22c55e"
                                  for v in df["d_e_ratio"].fillna(0)],
                ))
                fig.add_hline(y=2.0, line_dash="dash", line_color="#ef4444",
                              annotation_text="Seuil élevé (2x)")
                plotly_layout(fig, "Debt / Equity")
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3 : Scores ──
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if "z_score" in df.columns and df["z_score"].notna().any():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["year"], y=df["z_score"],
                    mode="lines+markers+text",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=8),
                ))
                fig.add_hrect(y0=2.99, y1=max(df["z_score"].max() * 1.1, 4),
                              fillcolor="#22c55e", opacity=0.05, annotation_text="Zone Sûre")
                fig.add_hrect(y0=1.81, y1=2.99,
                              fillcolor="#f59e0b", opacity=0.05, annotation_text="Zone Grise")
                fig.add_hrect(y0=min(df["z_score"].min() * 0.9, 0), y1=1.81,
                              fillcolor="#ef4444", opacity=0.05, annotation_text="Zone Danger")
                plotly_layout(fig, "Altman Z-Score")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "f_score" in df.columns and df["f_score"].notna().any():
                colors = ["#22c55e" if v >= 8 else "#f59e0b" if v >= 4 else "#ef4444"
                          for v in df["f_score"].fillna(0)]
                fig = go.Figure(go.Bar(x=df["year"], y=df["f_score"], marker_color=colors))
                fig.add_hline(y=8, line_dash="dash", line_color="#22c55e",
                              annotation_text="Solide ≥ 8")
                fig.add_hline(y=4, line_dash="dash", line_color="#f59e0b",
                              annotation_text="Neutre ≥ 4")
                plotly_layout(fig, "Piotroski F-Score")
                fig.update_yaxes(range=[0, 9.5])
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4 : Rentabilité ──
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for col_name, (label, color) in [
                ("roe", ("ROE", "#3b82f6")),
                ("roa", ("ROA", "#22c55e")),
            ]:
                if col_name in df.columns and df[col_name].notna().any():
                    fig.add_trace(go.Scatter(
                        x=df["year"], y=df[col_name] * 100,
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                    ))
            plotly_layout(fig, "ROE et ROA (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "health_score" in df.columns and df["health_score"].notna().any():
                fig = go.Figure(go.Scatter(
                    x=df["year"], y=df["health_score"],
                    mode="lines+markers",
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.1)",
                    line=dict(color="#3b82f6", width=2),
                ))
                plotly_layout(fig, "Score Santé Composite (0–100)")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)


def render_data_table(metrics_df: pd.DataFrame) -> None:
    """Affiche le tableau de données brutes et calculées."""
    section_header("🗃️ Données Brutes & Calculées")

    display_cols = [c for c in [
        "year", "revenue", "gross_profit", "net_income", "ebitda",
        "total_assets", "shareholders_equity", "total_debt",
        "net_margin", "roe", "roa",
        "current_ratio", "quick_ratio", "d_e_ratio",
        "z_score", "f_score", "health_score",
    ] if c in metrics_df.columns]

    display_df = metrics_df[display_cols].copy()

    # Formatage lisible
    pct_cols = ["net_margin", "roe", "roa", "gross_margin", "ebitda_margin"]
    for c in pct_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")

    ratio_cols = ["current_ratio", "quick_ratio", "d_e_ratio"]
    for c in ratio_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")

    large_cols = ["revenue", "gross_profit", "net_income", "ebitda", "total_assets",
                  "shareholders_equity", "total_debt"]
    for c in large_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(
                lambda x: format_large_number(x) if pd.notna(x) else "N/A"
            )

    st.dataframe(display_df.set_index("year"), use_container_width=True)


# ─────────────────────────────────────────────
# Mapping UI (pour fichiers uploadés)
# ─────────────────────────────────────────────

def render_mapping_ui(df_raw: pd.DataFrame, detected_mapping: dict) -> dict:
    """
    Affiche un formulaire permettant à l'utilisateur de confirmer ou corriger le mapping automatique.
    Retourne le mapping validé.
    """
    section_header("🗺️ Mapping des Colonnes")
    st.markdown("""
    <div class="warning-box">
        ⚠️ Le mapping automatique a été effectué. Vérifiez et corrigez si nécessaire.
    </div>
    """, unsafe_allow_html=True)

    available = ["(non mappé)"] + list(df_raw.columns)
    canonical_labels = {
        "revenue": "Chiffre d'Affaires (revenue)",
        "net_income": "Résultat Net (net_income)",
        "gross_profit": "Résultat Brut (gross_profit)",
        "ebitda": "EBITDA",
        "ebit": "EBIT / Rés. d'exploitation",
        "total_assets": "Total Actif",
        "current_assets": "Actif Courant",
        "inventories": "Stocks",
        "cash": "Trésorerie",
        "current_liabilities": "Passif Courant",
        "total_liabilities": "Passif Total",
        "total_debt": "Dette Totale",
        "shareholders_equity": "Capitaux Propres",
        "retained_earnings": "Résultats Reportés",
        "operating_cash_flow": "Cash Flow Opérationnel",
        "interest_expense": "Charges d'Intérêts",
        "shares_outstanding": "Actions en Circulation",
        "year": "Année",
    }

    confirmed_mapping: dict[str, Optional[str]] = {}
    cols = st.columns(2)
    for idx, (canonical, label) in enumerate(canonical_labels.items()):
        with cols[idx % 2]:
            current = detected_mapping.get(canonical)
            default_idx = available.index(current) if current and current in available else 0
            choice = st.selectbox(
                label=label,
                options=available,
                index=default_idx,
                key=f"map_{canonical}",
            )
            confirmed_mapping[canonical] = choice if choice != "(non mappé)" else None

    return confirmed_mapping


# ─────────────────────────────────────────────
# Application principale
# ─────────────────────────────────────────────

def main() -> None:
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px">
            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#e2e8f0">
                📊 FinHealth
            </div>
            <div style="font-size:11px;color:#64748b;margin-top:2px">
                Analyzer v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        data_source = st.radio(
            "Source de données",
            ["🔍 Ticker (yfinance)", "📂 Upload Fichier"],
            label_visibility="collapsed",
        )

        st.divider()

        currency = st.selectbox("Devise d'affichage", ["USD", "EUR", "GBP", "JPY"], index=0)

        st.divider()
        st.markdown("""
        <div style="font-size:10px;color:#334155;text-align:center;line-height:1.6">
            Ratios calculés selon les normes IFRS/GAAP.<br>
            Pour usage analytique uniquement.
        </div>
        """, unsafe_allow_html=True)

    # ── Titre principal ──
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
               background:linear-gradient(135deg,#3b82f6,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:4px">
        Analyse de Santé Financière
    </h1>
    <p style="color:#64748b;font-size:13px;margin-bottom:24px">
        ETL · Ratios · Altman Z-Score · Piotroski F-Score
    </p>
    """, unsafe_allow_html=True)

    # ─── Source : yfinance ───
    if "🔍" in data_source:
        col_search, col_btn = st.columns([3, 1])
        with col_search:
            ticker_input = st.text_input(
                "Ticker boursier",
                placeholder="Ex : AAPL, MSFT, TTE.PA, MC.PA ...",
                label_visibility="collapsed",
            )
        with col_btn:
            search_btn = st.button("Analyser", use_container_width=True)

        if search_btn and ticker_input:
            with st.spinner(f"Chargement des données pour **{ticker_input.upper()}**..."):
                try:
                    df_raw, company_info, market_data = load_from_yfinance(ticker_input)
                    st.session_state["df_raw"] = df_raw
                    st.session_state["company_info"] = company_info
                    st.session_state["market_data"] = market_data
                    st.session_state["mapping"] = None  # déjà canonique
                    st.session_state["currency"] = company_info.get("currency", currency)
                    st.session_state["source"] = "yfinance"
                except ValueError as e:
                    st.error(f"❌ {e}")
                    return

        if "df_raw" in st.session_state and st.session_state.get("source") == "yfinance":
            df_raw = st.session_state["df_raw"]
            company_info = st.session_state.get("company_info", {})
            market_data = st.session_state.get("market_data", {})
            disp_currency = st.session_state.get("currency", currency)

            # Info entreprise
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:16px;margin-bottom:16px">
                <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:#e2e8f0">
                    {company_info.get('name', ticker_input)}
                </div>
                <div style="margin-top:6px">
                    <span class="info-chip">{company_info.get('sector','N/A')}</span>
                    <span class="info-chip">{company_info.get('industry','N/A')}</span>
                    <span class="info-chip">{company_info.get('country','N/A')}</span>
                    <span class="info-chip">{disp_currency}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Calcul des métriques..."):
                metrics_df = compute_all_metrics(df_raw, market_data)

            _render_full_dashboard(metrics_df, disp_currency)

    # ─── Source : fichier ───
    else:
        uploaded = st.file_uploader(
            "Uploadez votre fichier financier",
            type=["csv", "xlsx", "xls"],
            help="Le fichier doit contenir une ligne par année avec les données financières.",
        )

        if uploaded is not None:
            try:
                df_raw, detected_mapping = load_from_file(uploaded)
            except ValueError as e:
                st.error(f"❌ {e}")
                return

            st.markdown(f"""
            <div class="success-box">
                ✅ Fichier chargé : {uploaded.name} — {len(df_raw)} lignes, {len(df_raw.columns)} colonnes
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Aperçu du fichier brut", expanded=False):
                st.dataframe(df_raw.head(10), use_container_width=True)

            confirmed_mapping = render_mapping_ui(df_raw, detected_mapping)

            if st.button("▶️ Lancer l'analyse", use_container_width=True):
                df_mapped = apply_mapping(df_raw, confirmed_mapping)
                df_mapped = clean_financial_df(df_mapped)

                if "year" not in df_mapped.columns:
                    df_mapped["year"] = range(
                        2024 - len(df_mapped) + 1, 2025
                    )

                with st.spinner("Calcul des métriques..."):
                    metrics_df = compute_all_metrics(df_mapped, {})

                _render_full_dashboard(metrics_df, currency)


def _render_full_dashboard(metrics_df: pd.DataFrame, currency: str) -> None:
    """Orchestre l'affichage complet du dashboard."""
    if metrics_df.empty:
        st.warning("Aucune donnée disponible pour construire le dashboard.")
        return

    latest = metrics_df.iloc[-1]
    prev = metrics_df.iloc[-2] if len(metrics_df) >= 2 else None

    # Nouvel indicateur d'Intelligence d'Analyse
    render_executive_summary(latest, prev)

    render_kpi_header(latest, currency)
    render_profitability(latest, prev)
    render_liquidity(latest, prev)
    render_solvency(latest, prev)
    render_scores(latest)
    render_piotroski_details(latest)
    render_charts(metrics_df, currency)
    render_data_table(metrics_df)


if __name__ == "__main__":
    main()
