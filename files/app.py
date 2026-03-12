"""
app.py
======
Interface Streamlit — orchestre data_handler.py et finance_metrics.py.
Lancer avec : streamlit run app.py
"""

from __future__ import annotations

import traceback
import re
import sys
import os
from typing import Optional

# Ensure that the 'files' directory is in the path to fix Streamlit Cloud ImportError
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_handler import (
    YFRateLimitError,
    apply_mapping,
    clean_financial_df,
    compute_working_capital,
    detect_column_mapping,
    format_large_number,
    format_percent,
    format_ratio,
    get_sector_averages,
    load_from_file,
    load_from_yfinance,
    load_news_from_yfinance,
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
    page_title="FinHealth Analyzer v2.1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design CSS personnalisé ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #020617; /* Deep Dark Mode Base */
        --bg-grad: radial-gradient(circle at top right, #0f172a, #020617);
        --surface: rgba(255, 255, 255, 0.02);
        --surface-hover: rgba(255, 255, 255, 0.04);
        --border: rgba(255, 255, 255, 0.1);
        --border-hover: rgba(255, 255, 255, 0.2);
        --accent: #38bdf8; /* Sleek Cyan/Blue */
        --accent-glow: rgba(56, 189, 248, 0.15);
        --text: #f8fafc;
        --muted: #94a3b8;
        --green: #10b981;
        --amber: #f59e0b;
        --red: #ef4444;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg) !important;
        background-image: var(--bg-grad) !important;
        color: var(--text) !important;
    }

    /* Shimmer Effect for Data Loading Feel */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    .stApp { 
        background-color: transparent !important; 
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(2, 6, 23, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.5);
    }

    .metric-card {
        background: var(--surface);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    /* Subtle inner shimmer line */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.03), transparent);
        transform: skewX(-20deg);
        transition: 0.7s;
    }
    .metric-card:hover::before {
        left: 150%;
    }

    .metric-card:hover { 
        background: var(--surface-hover);
        border-color: var(--border-hover); 
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px var(--accent-glow);
    }

    /* Info-bulles pur CSS */
    .tooltip-container {
        position: relative;
        display: inline-block;
        border-bottom: 1px dashed var(--muted);
        cursor: help;
    }
    
    .tooltip-container .tooltip-text {
        visibility: hidden;
        width: 240px;
        background-color: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(8px);
        color: var(--text);
        text-transform: none;
        letter-spacing: normal;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        font-size: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        position: absolute;
        z-index: 50;
        bottom: 130%;
        left: 50%;
        margin-left: -120px;
        opacity: 0;
        transition: opacity 0.3s, bottom 0.3s;
    }

    .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
        bottom: 140%;
    }

    .metric-label {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
    }
    .metric-delta {
        font-size: 13px;
        margin-top: 6px;
        font-weight: 500;
    }

    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid transparent;
    }

    .section-header {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--accent);
        margin: 32px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 8px var(--accent);
    }

    .info-chip {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 12px;
        display: inline-block;
        margin: 4px 4px 4px 0;
        color: var(--muted);
    }

    /* Buttons Premium Interaction */
    .stButton>button {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)) !important;
        backdrop-filter: blur(10px) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    .stButton>button:hover { 
        border-color: var(--accent) !important;
        background: rgba(56, 189, 248, 0.1) !important;
        color: var(--accent) !important;
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2), 0 0 15px var(--accent-glow) !important;
    }

    div[data-testid="metric-container"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 14px !important;
        font-weight: 500 !important;
        padding-bottom: 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom-color: var(--border) !important;
        margin-bottom: 24px !important;
    }

    /* Dataframes with Glassmorphism */
    .stDataFrame { 
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 8px !important;
    }

    .warning-box, .success-box, .danger-box {
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 14px;
        margin: 12px 0;
        border: 1px solid;
    }
    .warning-box {
        background: rgba(245, 158, 11, 0.05);
        border-color: rgba(245, 158, 11, 0.2);
        color: var(--amber);
    }
    .success-box {
        background: rgba(16, 185, 129, 0.05);
        border-color: rgba(16, 185, 129, 0.2);
        color: var(--green);
    }
    .danger-box {
        background: rgba(239, 68, 68, 0.05);
        border-color: rgba(239, 68, 68, 0.2);
        color: var(--red);
    }
    
    /* Custom Gradient Glows for Charts container */
    [data-testid="stPlotlyChart"] {
        position: relative;
        z-index: 1;
    }
    [data-testid="stPlotlyChart"]::before {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 70%; height: 70%;
        background: radial-gradient(circle, var(--accent-glow) 0%, transparent 60%);
        transform: translate(-50%, -50%);
        z-index: -1;
        pointer-events: none;
        filter: blur(60px);
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font_color": "#94a3b8",
    "gridcolor": "rgba(255,255,255,0.05)",
}


def plotly_layout(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    """Applique le thème ultra premium transparent à une figure Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=16, color="#f8fafc", weight="bold")),
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(color=PLOTLY_THEME["font_color"], family="Inter, sans-serif"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=20),
        xaxis=dict(gridcolor=PLOTLY_THEME["gridcolor"], showgrid=True, gridwidth=1, zeroline=False),
        yaxis=dict(gridcolor=PLOTLY_THEME["gridcolor"], showgrid=True, gridwidth=1, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_size=13, font_family="Inter"),
    )
    return fig


# ─────────────────────────────────────────────
# Calculs par ligne du DataFrame
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_explanation(metric_name: str, value: float, ticker: str) -> str:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key or api_key in ["VOTRE_CLE_API_GEMINI_ICI", "TO_BE_FILLED_BY_USER", ""]:
        return "Clé API intégrée ou locale requise pour l'analyse IA."
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"Tu es un analyste financier senior. Explique de manière ultra-concise (2-3 phrases) ce que signifie un {metric_name} de {value} pour l'entreprise {ticker}. Est-ce bon ou mauvais ?"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Erreur IA : {str(e)}"

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_sentiment_and_cashflow(ticker: str, news: list[dict], latest_metrics: dict) -> tuple[str, str, int]:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key or api_key in ["VOTRE_CLE_API_GEMINI_ICI", "TO_BE_FILLED_BY_USER", ""]:
        return "Clé API requise (secrets.toml ou env).", "Non analysé.", 50

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        news_text = "\n".join([f"- {n.get('title', '')}" for n in news[:10]])
        ni = format_large_number(latest_metrics.get("net_income"))
        ocf = format_large_number(latest_metrics.get("operating_cash_flow"))
        
        prompt = f"""
Tu es un analyste quantitatif senior.
1. Lis ces titres d'actualité récents pour le ticker {ticker} :
{news_text}
Génère un résumé macro ultra-concis en 3 petits bullet points (Bullish vs Bearish) et donne un SCORE de sentiment de 0 (Très Bearish) à 100 (Très Bullish).

2. Analyse ces métriques pour détecter une potentielle anomalie :
- Résultat Net : {ni}
- Cash Flow Opérationnel (CFO) : {ocf}
Le Cash Flow Opérationnel est-il cohérent par rapport au résultat net ? (Réponds en 1 à 2 phrases max, sois ultra-direct).

Format strict demandé (respecte exactement la casse de ces balises) :
SCORE: <ton score entre 0 et 100>
SENTIMENT:
<tes 3 bullet points ici>
ANOMALIE_CF:
<ton analyse du cash flow ici>
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        ).text
        
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else 50
        
        sent_match = re.search(r"SENTIMENT:(.*?)(?=ANOMALIE_CF:|$)", response, re.DOTALL)
        sentiment_text = sent_match.group(1).strip() if sent_match else "Analyse indisponible."
        
        cf_match = re.search(r"ANOMALIE_CF:(.*?)$", response, re.DOTALL)
        cf_anomaly = cf_match.group(1).strip() if cf_match else "Analyse indisponible."
        
        return sentiment_text, cf_anomaly, score
    except Exception as e:
        return f"Erreur IA : {str(e)}", "Erreur réseau", 50


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
    section_header("Key Performance Indicators")
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
    section_header("Executive Summary")
    
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
    section_header("Profitability Analysis")
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
    section_header("Liquidity Metrics")
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
    section_header("Solvency & Leverage")
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
    section_header("Advanced Health Scoring")

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
                number={"font": {"color": hs_color, "family": "Inter", "weight": "bold"}},
            ))
            fig = plotly_layout(fig, height=220)
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
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

    section_header("Détail Piotroski F-Score (Radar)")
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
    section_header("Évolution Temporelle")

    df = metrics_df.sort_values("year")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Revenus & Marges", "Liquidité & Levier",
        "Scores", "Rentabilité", "Comparaison & Dividendes"
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
                
                # --- AI Button ---
                if st.button("💡 Analyse IA Z-Score", key="btn_ai_zscore"):
                    latest_z = df["z_score"].iloc[-1]
                    ticker = st.session_state.get("company_info", {}).get("name", "l'entreprise")
                    with st.spinner("Analyse IA..."):
                        expl = generate_ai_explanation("Altman Z-Score", latest_z, ticker)
                    st.info(expl)

                # --- WHAT IF SCENARIO ---
                st.markdown("##### 🧪 Scénario 'What-If' (Stress Test)")
                drop_pct = st.slider("Baisse simulée du Chiffre d'Affaires (%)", min_value=0, max_value=20, value=0, step=5)
                
                if drop_pct > 0 and 'df_raw' in st.session_state:
                    latest_raw = st.session_state['df_raw'].iloc[-1]
                    mcap = st.session_state.get('market_data', {}).get(latest_raw['year'], {}).get('market_cap', 0)
                    if not pd.isna(mcap) and mcap > 0:
                        original_revenue = latest_raw.get("revenue", 0)
                        sim_revenue = original_revenue * (1 - (drop_pct / 100))
                        
                        # Recalculate z-score
                        from finance_metrics import altman_z_score
                        wc = compute_working_capital(latest_raw)
                        ret_earn = latest_raw.get("retained_earnings", 0)
                        ebit = latest_raw.get("ebit", 0)
                        tot_liab = latest_raw.get("total_liabilities", 0)
                        tot_assets = latest_raw.get("total_assets", 0)
                        
                        sim_z = altman_z_score(wc, ret_earn, ebit, mcap, tot_liab, sim_revenue, tot_assets)
                        latest_z = df["z_score"].iloc[-1]
                        
                        if sim_z is not None:
                            delta = sim_z - latest_z
                            st.metric(label="Z-Score Simulé", value=f"{sim_z:.2f}", delta=f"{delta:.2f}")

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
            
            # --- AI Button ---
            if st.button("💡 Analyse IA Rentabilité (ROE)", key="btn_ai_roe"):
                latest_roe = df["roe"].iloc[-1] * 100 if "roe" in df.columns else 0
                ticker = st.session_state.get("company_info", {}).get("name", "l'entreprise")
                with st.spinner("Analyse IA..."):
                    expl = generate_ai_explanation("ROE (Return on Equity)", latest_roe, ticker)
                st.info(expl)

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

    # ── Tab 5 : Comparaison & Dividendes ──
    with tab5:
        col1, col2 = st.columns(2)
        company_info = st.session_state.get("company_info", {})
        sector = company_info.get("sector", "Industrials")
        
        with col1:
            st.markdown("##### 🏢 Comparaison Sectorielle")
            st.caption(f"Secteur de référence : {sector}")
            sector_avg = get_sector_averages(sector)
            
            latest = df.iloc[-1] if not df.empty else pd.Series()
            
            categories = ['Marge Brute', 'Marge EBITDA', 'Marge Nette']
            ticker_vals = [
                latest.get("gross_margin", 0) * 100,
                latest.get("ebitda_margin", 0) * 100,
                latest.get("net_margin", 0) * 100
            ]
            sector_vals = [
                sector_avg.get("gross_profit_margin", 0) * 100,
                sector_avg.get("ebitda_margin", 0) * 100,
                sector_avg.get("net_profit_margin", 0) * 100
            ]
            
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=categories, y=ticker_vals, name=company_info.get("name", "Ticker"), marker_color="#38bdf8"
            ))
            fig_cmp.add_trace(go.Bar(
                x=categories, y=sector_vals, name=f"Moy. {sector}", marker_color="#475569"
            ))
            fig_cmp.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', family='Inter'),
                margin=dict(l=20, r=20, t=30, b=20),
                height=300
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            
        with col2:
            st.markdown("##### 💸 Simulateur de Dividendes")
            yield_pct = company_info.get("dividend_yield", 0.0)
            if yield_pct and yield_pct > 0:
                st.info(f"Rendement actuel estimé : **{yield_pct*100:.2f}%**")
                invested = st.number_input("Montant investi (Simulé)", min_value=100, max_value=10_000_000, value=10000, step=1000)
                annual_div = invested * yield_pct
                
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;text-align:center;margin-top:16px;">
                    <div style="font-size:12px;color:var(--muted);text-transform:uppercase;">Dividendes annuels projetés</div>
                    <div style="font-size:32px;font-weight:700;color:var(--green);">{annual_div:,.2f} {currency}</div>
                    <div style="font-size:11px;color:var(--muted);margin-top:4px;">Basé sur le rendement historique et le prix actuel</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("L'entreprise ne verse pas de dividendes ou les données historiques sont indisponibles.")


def render_data_table(metrics_df: pd.DataFrame) -> None:
    """Affiche le tableau de données brutes et calculées."""
    section_header("Données Brutes & Calculées")

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
    section_header("Mapping des Colonnes")
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

def render_landing_page() -> None:
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    st.title("🚀 Bienvenue sur FinHealth Analyzer")
    st.markdown("""
        <p style="color:var(--muted); font-size:16px; margin-bottom: 32px;">
            L'outil d'audit de santé financière propulsé par l'IA. 
            Découvrez des insights immédiats et auditez la solvabilité des entreprises de votre choix.
        </p>
    """, unsafe_allow_html=True)

    section_header("Quick Start : Exemples")
    
    col1, col2, col3 = st.columns(3)
    
    def set_ticker(t: str):
        st.session_state["analyze_ticker"] = t

    with col1:
        st.button("Apple (AAPL)", use_container_width=True, on_click=set_ticker, args=("AAPL",))
    with col2:
        st.button("NVIDIA (NVDA)", use_container_width=True, on_click=set_ticker, args=("NVDA",))
    with col3:
        st.button("LVMH (MC.PA)", use_container_width=True, on_click=set_ticker, args=("MC.PA",))

    st.markdown("<div style='margin-top: 48px;'></div>", unsafe_allow_html=True)
    section_header("Guide de Fonctionnalités")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 100%;">
            <div style="font-size: 32px; margin-bottom: 12px;">📊</div>
            <div style="font-weight: 600; margin-bottom: 8px; color: var(--text);">Dashboards Interactifs</div>
            <div style="font-size: 13px; color: var(--muted);">Visualisez l'évolution temporelle des marges, de la croissance et de la liquidité avec des graphiques dynamiques.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 100%;">
            <div style="font-size: 32px; margin-bottom: 12px;">🛡️</div>
            <div style="font-weight: 600; margin-bottom: 8px; color: var(--text);">Scores de Solvabilité</div>
            <div style="font-size: 13px; color: var(--muted);">Calcul automatique du Piotroski F-Score et de l'Altman Z-Score pour prévenir les risques de faillite.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 100%;">
            <div style="font-size: 32px; margin-bottom: 12px;">💬</div>
            <div style="font-weight: 600; margin-bottom: 8px; color: var(--text);">Assistant IA Financier</div>
            <div style="font-size: 13px; color: var(--muted);">Interrogez un analyste virtuel (Gemini) directement intégré pour interpréter les résultats et ratios.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 48px;'></div>", unsafe_allow_html=True)
    st.info("💡 **Instruction :** Entrez un ticker dans la barre de recherche en haut pour commencer.")


def main() -> None:
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px">
            <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:#e2e8f0;display:flex;align-items:center;justify-content:center;gap:8px;">
                <span style="font-size:28px;">📈</span> FinHealth
            </div>
            <div style="font-size:11px;color:#64748b;margin-top:2px">
                Analyzer v2.1
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        
        st.markdown("### 🛠️ Configuration")

        data_source = st.radio(
            "Source de données",
            ["Ticker (yfinance)", "Upload Fichier"],
            label_visibility="collapsed",
        )

        st.divider()

        currency = st.selectbox("Devise d'affichage", ["USD", "EUR", "GBP", "JPY"], index=0)

        st.divider()

        # Indicateur IA
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY")
            
        if api_key and api_key not in ["VOTRE_CLE_API_GEMINI_ICI", "TO_BE_FILLED_BY_USER", ""]:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;background:rgba(16, 185, 129, 0.1);border:1px solid rgba(16, 185, 129, 0.3);border-radius:8px;padding:8px 12px;margin-bottom:16px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981;"></div>
                <span style="color:#10b981;font-size:13px;font-weight:600;">Assistant IA : Connecté</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;background:rgba(239, 68, 68, 0.1);border:1px solid rgba(239, 68, 68, 0.3);border-radius:8px;padding:8px 12px;margin-bottom:16px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px #ef4444;"></div>
                <span style="color:#ef4444;font-size:13px;font-weight:600;">Assistant IA : Déconnecté</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div style="font-size:12px;color:#e2e8f0;line-height:1.6;margin-bottom:8px">
            <strong>📖 Méthodologie</strong><br>
            <span style="color:#94a3b8;font-size:11px;">
            • <strong>Altman Z-Score</strong> : Prédit la probabilité de faillite à 2 ans via des ratios de liquidité et rentabilité.<br>
            • <strong>Piotroski F-Score</strong> : Évalue la solidité et la tendance financière sur 9 critères stricts (Score 0-9).
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div style="font-size:12px;color:#94a3b8;line-height:1.6;margin-bottom:8px">
            <strong>À propos</strong><br>
            Données de marché par <a href="https://finance.yahoo.com/" target="_blank" style="color:#38bdf8;text-decoration:none;">Yahoo Finance</a>.<br>
            Analyse propulsée par <strong style="color:#e2e8f0;">Gemini 2.5 Flash</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:10px;color:#334155;text-align:center;line-height:1.6;margin-top:24px;">
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
    if "Ticker" in data_source:
        default_ticker = ""
        do_auto_search = False
        if st.session_state.get("analyze_ticker"):
            default_ticker = st.session_state["analyze_ticker"]
            do_auto_search = True
            st.session_state["analyze_ticker"] = ""
            
        ticker_input = st.text_input(
            "Ticker boursier",
            value=default_ticker,
            placeholder="Rechercher un Ticker (ex: AAPL) et appuyer sur Entrée...",
            label_visibility="collapsed",
        )

        # L'appui sur 'Enter' lance un rerun Streamlit avec la nouvelle valeur de ticker_input
        # On exécute l'analyse s'il y a une recherche auto (landing page)
        # OU si le ticker a été modifié (appui sur Enter)
        trigger_analysis = do_auto_search or (ticker_input and ticker_input != st.session_state.get("last_analyzed_ticker", ""))

        if trigger_analysis and ticker_input:
            st.session_state["last_analyzed_ticker"] = ticker_input
            with st.spinner(f"Chargement des données pour **{ticker_input.upper()}**..."):
                try:
                    df_raw, company_info, market_data = load_from_yfinance(ticker_input)
                    st.session_state["df_raw"] = df_raw
                    st.session_state["company_info"] = company_info
                    st.session_state["market_data"] = market_data
                    st.session_state["mapping"] = None  # déjà canonique
                    st.session_state["currency"] = company_info.get("currency", currency)
                    st.session_state["source"] = "yfinance"
                except YFRateLimitError as e:
                    st.warning(f"⚠️ {e}")
                    st.session_state.pop("df_raw", None)
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.session_state.pop("df_raw", None)

        query = "df_raw" in st.session_state and st.session_state.get("source") == "yfinance"

        if query:
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

            # --- MODULE 1 : INTELLIGENCE & SENTIMENT (SIDEBAR) ---
            with st.spinner("Analyse IA en cours (Sentiment & Cash Flow)..."):
                news = load_news_from_yfinance(ticker_input, limit=10)
                latest_metrics = metrics_df.iloc[-1].to_dict() if len(metrics_df) > 0 else {}
                sentiment_text, cf_anomaly, score = analyze_sentiment_and_cashflow(ticker_input, news, latest_metrics)
                
            st.sidebar.markdown("### 🧠 Intelligence & Sentiment")
            
            score_color = "#22c55e" if score >= 60 else ("#f59e0b" if score >= 40 else "#ef4444")
            score_label = "Bullish 📈" if score >= 60 else ("Neutre 😐" if score >= 40 else "Bearish 📉")
            
            st.sidebar.markdown(f"""
<div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:12px;margin-bottom:10px;">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Index de Sentiment</div>
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="font-size:22px;font-weight:700;color:{score_color};">{score}/100</div>
    <div style="flex-grow:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">
      <div style="height:100%;width:{score}%;background:{score_color};"></div>
    </div>
    <div style="font-size:11px;color:{score_color};font-weight:600;">{score_label}</div>
  </div>
</div>
""", unsafe_allow_html=True)

            with st.sidebar.expander("📰 Résumé Actualités", expanded=True):
                st.markdown(sentiment_text)

            with st.sidebar.expander("💧 Anomalie Cash Flow", expanded=False):
                st.markdown(cf_anomaly)
            # --------------------------------------------------------

            _render_full_dashboard(metrics_df, disp_currency)
        else:
            render_landing_page()

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


def render_ai_assistant_tab(metrics_df: pd.DataFrame, company_info: dict) -> None:
    """Affiche le chat de l'Assistant IA basé sur la nouvelle API google-genai."""
    st.markdown("### Assistant Financier")
    st.markdown("Posez vos questions sur la santé financière de l'entreprise étudiée. L'IA a accès à tous les ratios calculés.")

    # Vérification de la clé API
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
        
    if not api_key or api_key == "VOTRE_CLE_API_GEMINI_ICI" or api_key == "TO_BE_FILLED_BY_USER":
        st.info("ℹ️ La clé API Gemini n'est pas configurée. Veuillez l'ajouter dans vos secrets Streamlit (`.streamlit/secrets.toml` en local ou Streamlit Cloud Secrets) pour activer l'assistant.")
        return

    # Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Bouton pour effacer l'historique
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("Effacer l'historique", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Configuration du modèle et du contexte initial
    latest = metrics_df.iloc[-1]
    company_name = company_info.get("name", "l'entreprise")
    sector = company_info.get("sector", "Non spécifié")
    
    def _safe_fmt(val, fmt_func=None, suffix=""):
        """Safely format a value, returning 'N/A' for None/NaN."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        if fmt_func:
            return fmt_func(val)
        return f"{val}{suffix}"

    context = f"""
    Tu es un Analyste Financier Senior. Ton rôle est d'analyser les résultats d'une entreprise de manière pédagogique, structurée et précise, exclusivement en français.
    Tu aides un utilisateur à comprendre la santé financière de l'entreprise {company_name} (Secteur: {sector}).
    Voici les dernières métriques clés (Année {latest.get('year', 'N/A')}) :
    - Chiffre d'Affaires : {_safe_fmt(latest.get('revenue'), format_large_number)}
    - Résultat Net : {_safe_fmt(latest.get('net_income'), format_large_number)}
    - Marge Nette : {_safe_fmt(latest.get('net_margin'), format_percent)}
    - ROE : {_safe_fmt(latest.get('roe'), format_percent)}
    - ROA : {_safe_fmt(latest.get('roa'), format_percent)}
    - Current Ratio (Liquidité) : {_safe_fmt(latest.get('current_ratio'), format_ratio)}
    - Debt/Equity (Levier) : {_safe_fmt(latest.get('d_e_ratio'), format_ratio)}
    - Altman Z-Score : {_safe_fmt(latest.get('z_score'), lambda v: f'{v:.2f}')}
    - Piotroski F-Score : {_safe_fmt(latest.get('f_score'), lambda v: f'{int(v)} / 9')}
    - Score Santé Global FiHealth : {_safe_fmt(latest.get('health_score'), lambda v: f'{v:.1f} / 100')}
    Réponds de manière concise et experte aux questions de l'utilisateur. Ne donne pas de conseils d'investissement garantis, garde un ton analytique.
    """

    # Affichage des messages passés
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez une question sur les résultats (ex: Que penser de la liquidité ?)..."):
        # Affichage du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Génération de la réponse de l'IA
        with st.chat_message("assistant"):
            try:
                from google import genai
                from google.genai import types
                
                client = genai.Client(api_key=api_key)
                
                # Formatage de l'historique
                gemini_history = []
                for msg in st.session_state.messages[:-1]:
                    gemini_role = "model" if msg["role"] == "assistant" else "user"
                    gemini_history.append(
                        types.Content(role=gemini_role, parts=[types.Part.from_text(text=msg["content"])])
                    )
                
                config = types.GenerateContentConfig(
                    system_instruction=context
                )
                
                chat = client.chats.create(
                    model="gemini-2.5-flash",
                    config=config,
                    history=gemini_history
                )
                
                response = chat.send_message(prompt)
                
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {str(e)}")



def _render_full_dashboard(metrics_df: pd.DataFrame, currency: str) -> None:
    """Orchestre l'affichage complet du dashboard."""
    if metrics_df.empty:
        st.warning("Aucune donnée disponible pour construire le dashboard.")
        return

    company_info = st.session_state.get("company_info", {})

    tab_dash, tab_ai = st.tabs(["Dashboard", "Assistant"])

    with tab_dash:
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

    with tab_ai:
        render_ai_assistant_tab(metrics_df, company_info)


if __name__ == "__main__":
    main()
