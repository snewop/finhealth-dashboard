"""
data_handler.py
===============
Couche d'import, nettoyage et mapping des données financières.
Supporte : fichiers Excel/CSV uploadés (avec mapping dynamique) et l'API yfinance.
"""

from __future__ import annotations

import re
import time
import pandas as pd
import streamlit as st
import yfinance as yf


class YFRateLimitError(ValueError):
    """Exception spécifique pour les blocages d'IP (Ex: Streamlit Cloud)."""
    pass


# ─────────────────────────────────────────────
# Dictionnaire de mapping (synonymes → noms canoniques)
# ─────────────────────────────────────────────

COLUMN_SYNONYMS: dict[str, list[str]] = {
    "revenue": [
        "revenue", "chiffre d'affaires", "ca", "ventes", "sales", "turnover",
        "total revenue", "net revenue", "produits", "revenus",
    ],
    "net_income": [
        "net income", "resultat net", "bénéfice net", "profit net", "net profit",
        "net earnings", "résultat", "income",
    ],
    "gross_profit": [
        "gross profit", "résultat brut", "marge brute", "gross margin value",
        "profit brut", "bénéfice brut",
    ],
    "ebitda": [
        "ebitda", "excédent brut d'exploitation", "ebe",
        "earnings before interest taxes depreciation amortization",
    ],
    "ebit": [
        "ebit", "résultat d'exploitation", "operating income", "operating profit",
        "résultat opérationnel",
    ],
    "total_assets": [
        "total assets", "total actif", "actif total", "total de l'actif",
        "assets",
    ],
    "current_assets": [
        "current assets", "actif courant", "actif circulant",
        "actifs courants",
    ],
    "inventories": [
        "inventory", "inventories", "stocks", "marchandises",
    ],
    "cash": [
        "cash", "cash and cash equivalents", "trésorerie",
        "liquidités", "disponibilités",
    ],
    "current_liabilities": [
        "current liabilities", "passif courant", "dettes courantes",
        "dettes à court terme",
    ],
    "total_liabilities": [
        "total liabilities", "passif total", "total des dettes",
        "dettes totales", "liabilities",
    ],
    "total_debt": [
        "total debt", "dette totale", "long term debt", "dette long terme",
        "borrowings", "emprunts",
    ],
    "shareholders_equity": [
        "shareholders equity", "capitaux propres", "equity",
        "total equity", "stockholders equity", "fonds propres",
    ],
    "retained_earnings": [
        "retained earnings", "résultats reportés", "réserves",
        "bénéfices non distribués",
    ],
    "operating_cash_flow": [
        "operating cash flow", "cash flow from operations",
        "flux de trésorerie opérationnel", "cfo",
    ],
    "interest_expense": [
        "interest expense", "charges financières", "charges d'intérêts",
        "intérêts", "financial charges",
    ],
    "market_cap": [
        "market cap", "capitalisation boursière", "market capitalization",
        "cap boursière",
    ],
    "shares_outstanding": [
        "shares outstanding", "actions en circulation", "nombre d'actions",
        "shares", "diluted shares",
    ],
    "working_capital": [
        "working capital", "fonds de roulement", "besoin en fonds de roulement",
    ],
}


def _normalize(text: str) -> str:
    """Normalise un nom de colonne pour la comparaison (minuscules, sans accents, sans ponctuation)."""
    text = text.lower().strip()
    text = re.sub(r"[àáâä]", "a", text)
    text = re.sub(r"[èéêë]", "e", text)
    text = re.sub(r"[ìíîï]", "i", text)
    text = re.sub(r"[òóôö]", "o", text)
    text = re.sub(r"[ùúûü]", "u", text)
    text = re.sub(r"[ç]", "c", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


import difflib

def detect_column_mapping(columns: list[str]) -> dict[str, Optional[str]]:
    """
    Détecte automatiquement les colonnes d'un DataFrame et les mappe aux noms canoniques.
    Utilise difflib pour être robuste contre les petites fautes de frappes.

    Retourne un dictionnaire {nom_canonique: colonne_détectée_ou_None}.
    """
    norm_cols = {_normalize(c): c for c in columns}
    mapping: dict[str, Optional[str]] = {}

    for canonical, synonyms in COLUMN_SYNONYMS.items():
        found = None
        # 1. Recherche exacte (normalisée)
        for syn in synonyms:
            norm_syn = _normalize(syn)
            if norm_syn in norm_cols:
                found = norm_cols[norm_syn]
                break
        
        # 2. Recherche par proximité si non trouvé (fautes de frappe)
        if found is None:
            for syn in synonyms:
                norm_syn = _normalize(syn)
                matches = difflib.get_close_matches(norm_syn, list(norm_cols.keys()), n=1, cutoff=0.8)
                if matches:
                    found = norm_cols[matches[0]]
                    break

        mapping[canonical] = found

    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Renomme les colonnes d'un DataFrame selon le mapping {canonique: original}.
    Seules les colonnes mappées sont conservées dans le résultat.
    """
    rename_map = {v: k for k, v in mapping.items() if v is not None}
    df_renamed = df.rename(columns=rename_map)
    canonical_cols = [k for k, v in mapping.items() if v is not None]
    return df_renamed[[c for c in canonical_cols if c in df_renamed.columns]]


# ─────────────────────────────────────────────
# Nettoyage
# ─────────────────────────────────────────────

def clean_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame financier :
    - Supprime les colonnes 100 % vides
    - Convertit les colonnes numériques (gère les chaînes comme "1,234.56")
    - Remplace les valeurs infinies par NaN
    """
    df = df.copy()

    for col in df.columns:
        if col == "year":
            continue
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[,$€£ ]", "", regex=True)
                .str.replace(",", ".")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    return df


def load_from_file(uploaded_file) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    """
    Charge un fichier Excel ou CSV uploadé via Streamlit (st.file_uploader).

    Retourne (DataFrame nettoyé, mapping détecté).
    Lève une ValueError si le format n'est pas supporté.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Format non supporté : {name}. Utilisez CSV ou Excel.")

    mapping = detect_column_mapping(df.columns.tolist())
    df_clean = clean_financial_df(df)
    return df_clean, mapping


# ─────────────────────────────────────────────
# Import via yfinance
# ─────────────────────────────────────────────

def _yfin_series_to_row(series: pd.Series, year: int) -> dict:
    """Transforme une Series yfinance (index = noms de métriques) en dict avec year."""
    row = {"year": year}
    for k, v in series.items():
        row[str(k)] = v
    return row


@st.cache_data(ttl=3600, show_spinner=False)
def load_from_yfinance(ticker: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Récupère les états financiers annuels pour le ticker donné via yfinance.

    Retourne :
    - DataFrame financier multi-années (index = année)
    - dict d'informations générales (nom, secteur, etc.)
    - dict avec market_cap et shares_outstanding par année

    Lève une ValueError si le ticker est introuvable ou les données vides.
    """
    try:
        # Retry mechanism for Too Many Requests (429)
        max_retries = 3
        last_exception = None
        for attempt in range(max_retries):
            try:
                ticker_obj = yf.Ticker(ticker.upper().strip())
                info = ticker_obj.info
                break  # Success
            except Exception as e:
                last_exception = e
                if '429' in str(e) or 'Rate limited' in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise YFRateLimitError(
                            "Yahoo Finance bloque temporairement l'IP des serveurs virtuels de Streamlit (Erreur 429). "
                            "Pour analyser cette entreprise, veuillez utiliser l'option 'Upload Fichier' avec vos données, "
                            "ou lancez l'application en local où cela fonctionne parfaitement."
                        ) from e
                else:
                    raise

        # Vérification basique que le ticker existe
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Certains tickers valides n'ont pas de prix en temps réel (ETF etc.)
            # On tente quand même de récupérer les statements
            pass

        income_stmt = ticker_obj.financials      # Compte de résultat (cols = dates)
        balance_sheet = ticker_obj.balance_sheet  # Bilan
        cash_flow = ticker_obj.cashflow           # Flux de trésorerie

        if income_stmt is None or income_stmt.empty:
            raise ValueError(f"Aucune donnée financière trouvée pour le ticker '{ticker}'.")

        rows = []
        market_data: dict[int, dict] = {}

        for date_col in income_stmt.columns:
            year = date_col.year
            row = {"year": year}

            def get(stmt: pd.DataFrame, *keys: str) -> Optional[float]:
                """Cherche la première clé disponible dans un statement."""
                for k in keys:
                    try:
                        val = stmt.at[k, date_col]
                        if pd.notna(val):
                            return float(val)
                    except (KeyError, TypeError):
                        pass
                return None

            # ── Compte de résultat ──
            row["revenue"] = get(income_stmt, "Total Revenue", "Operating Revenue")
            row["gross_profit"] = get(income_stmt, "Gross Profit", "Total Revenue")
            row["ebit"] = get(income_stmt, "EBIT", "Operating Income")
            row["ebitda"] = get(income_stmt, "EBITDA", "Normalized EBITDA")
            row["net_income"] = get(income_stmt, "Net Income Common Stockholders", "Net Income")
            row["interest_expense"] = get(income_stmt, "Interest Expense")

            # ── Bilan ──
            row["total_assets"] = get(balance_sheet, "Total Assets")
            row["current_assets"] = get(balance_sheet, "Current Assets")
            row["inventories"] = get(balance_sheet, "Inventory")
            row["cash"] = get(balance_sheet, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial")
            row["current_liabilities"] = get(balance_sheet, "Current Liabilities")
            row["total_liabilities"] = get(balance_sheet, "Total Liabilities Net Minority Interest", "Total Liabilities", "Total Non Current Liabilities Net Minority Interest")
            row["total_debt"] = get(balance_sheet, "Total Debt", "Long Term Debt", "Net Debt")
            row["shareholders_equity"] = get(balance_sheet, "Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity")
            row["retained_earnings"] = get(balance_sheet, "Retained Earnings")
            row["shares_outstanding"] = get(balance_sheet, "Ordinary Shares Number", "Basic Average Shares")

            # ── Flux de trésorerie ──
            row["operating_cash_flow"] = get(cash_flow, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")

            # ── Métriques de marché (snapshot actuel) ──
            market_cap = info.get("marketCap")
            market_data[year] = {
                "market_cap": float(market_cap) if market_cap else None,
                "shares_outstanding": info.get("sharesOutstanding"),
            }

            rows.append(row)

        df = pd.DataFrame(rows).sort_values("year", ascending=True).reset_index(drop=True)
        df = clean_financial_df(df)

        company_info = {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "currency": info.get("financialCurrency", "USD"),
            "description": info.get("longBusinessSummary", ""),
            "website": info.get("website", ""),
            "logo": info.get("logo_url", ""),
        }

        return df, company_info, market_data

    except Exception as exc:
        raise ValueError(f"Erreur lors du chargement de '{ticker}' : {exc}") from exc


# ─────────────────────────────────────────────
# Utilitaires de préparation pour le moteur de calcul
# ─────────────────────────────────────────────

def safe_get(row: pd.Series, col: str) -> Optional[float]:
    """Récupère une valeur d'une ligne pandas, retourne None si absente ou NaN."""
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return float(val)


def compute_working_capital(row: pd.Series) -> Optional[float]:
    """Calcule le fonds de roulement = Actif courant - Passif courant."""
    ca = safe_get(row, "current_assets")
    cl = safe_get(row, "current_liabilities")
    if ca is None or cl is None:
        return None
    return ca - cl


def format_large_number(value: Optional[float], currency: str = "USD") -> str:
    """
    Formate un grand nombre financier en forme lisible.
    Ex : 1_234_567_890 → "1.23 Mrd $"
    """
    if value is None or pd.isna(value):
        return "N/A"

    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    sym = symbols.get(currency, currency)

    abs_v = abs(value)
    sign = "-" if value < 0 else ""

    if abs_v >= 1e12:
        return f"{sign}{abs_v / 1e12:.2f} T{sym}"
    elif abs_v >= 1e9:
        return f"{sign}{abs_v / 1e9:.2f} Mrd{sym}"
    elif abs_v >= 1e6:
        return f"{sign}{abs_v / 1e6:.2f} M{sym}"
    elif abs_v >= 1e3:
        return f"{sign}{abs_v / 1e3:.2f} K{sym}"
    else:
        return f"{sign}{abs_v:.2f} {sym}"


def format_percent(value: Optional[float], decimals: int = 1) -> str:
    """Formate un ratio décimal en pourcentage lisible. Ex : 0.2345 → '23.5 %'"""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f} %"


def format_ratio(value: Optional[float], decimals: int = 2) -> str:
    """Formate un ratio. Ex : 1.456 → '1.46x'"""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}x"
