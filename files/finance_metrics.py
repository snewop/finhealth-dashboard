"""
finance_metrics.py
==================
Moteur de calcul financier pur.
Toutes les fonctions sont sans effets de bord : elles prennent des nombres et retournent des nombres.
"""

from __future__ import annotations
from typing import Optional


import pandas as pd

# ─────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────

def _safe_divide(numerator: float, denominator: float, default: Optional[float] = None) -> Optional[float]:
    """Division sécurisée : retourne `default` si le dénominateur est nul ou None."""
    if pd.isna(denominator) or denominator == 0:
        return default
    if pd.isna(numerator):
        return default
    return numerator / denominator


# ─────────────────────────────────────────────
# Ratios de Rentabilité
# ─────────────────────────────────────────────

def net_profit_margin(net_income: float, revenue: float) -> Optional[float]:
    """
    Marge nette = Résultat net / Chiffre d'affaires.
    Indique quelle part de chaque euro de CA devient du profit.
    """
    return _safe_divide(net_income, revenue)


def return_on_equity(net_income: float, shareholders_equity: float) -> Optional[float]:
    """
    ROE = Résultat net / Capitaux propres.
    Mesure l'efficacité avec laquelle l'entreprise génère du profit à partir des fonds propres.
    """
    return _safe_divide(net_income, shareholders_equity)


def return_on_assets(net_income: float, total_assets: float) -> Optional[float]:
    """
    ROA = Résultat net / Total actif.
    Mesure l'efficacité d'utilisation des actifs pour générer du profit.
    """
    return _safe_divide(net_income, total_assets)


def gross_profit_margin(gross_profit: float, revenue: float) -> Optional[float]:
    """
    Marge brute = Résultat brut / CA.
    """
    return _safe_divide(gross_profit, revenue)


def ebitda_margin(ebitda: float, revenue: float) -> Optional[float]:
    """
    Marge EBITDA = EBITDA / CA.
    """
    return _safe_divide(ebitda, revenue)


# ─────────────────────────────────────────────
# Ratios de Liquidité
# ─────────────────────────────────────────────

def current_ratio(current_assets: float, current_liabilities: float) -> Optional[float]:
    """
    Ratio de liquidité générale = Actif courant / Passif courant.
    > 1 : l'entreprise peut couvrir ses dettes à court terme.
    """
    return _safe_divide(current_assets, current_liabilities)


def quick_ratio(current_assets: float, inventories: float, current_liabilities: float) -> Optional[float]:
    """
    Ratio de liquidité immédiate = (Actif courant - Stocks) / Passif courant.
    Plus conservateur que le current ratio.
    """
    if current_assets is None or inventories is None:
        return None
    return _safe_divide(current_assets - inventories, current_liabilities)


def cash_ratio(cash: float, current_liabilities: float) -> Optional[float]:
    """
    Ratio de trésorerie = Trésorerie / Passif courant.
    """
    return _safe_divide(cash, current_liabilities)


# ─────────────────────────────────────────────
# Ratios de Solvabilité / Levier
# ─────────────────────────────────────────────

def debt_to_equity(total_debt: float, shareholders_equity: float) -> Optional[float]:
    """
    D/E = Dette totale / Capitaux propres.
    Mesure le levier financier. > 2 peut indiquer un risque élevé.
    """
    return _safe_divide(total_debt, shareholders_equity)


def debt_to_assets(total_debt: float, total_assets: float) -> Optional[float]:
    """
    Ratio d'endettement = Dette totale / Total actif.
    """
    return _safe_divide(total_debt, total_assets)


def interest_coverage(ebit: float, interest_expense: float) -> Optional[float]:
    """
    Couverture des intérêts = EBIT / Charges d'intérêts.
    > 3 est généralement considéré comme sain.
    """
    return _safe_divide(ebit, interest_expense)


# ─────────────────────────────────────────────
# Altman Z-Score (modèle 1968, entreprises cotées)
# ─────────────────────────────────────────────

def altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
    total_assets: float,
) -> Optional[float]:
    """
    Altman Z-Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Interprétation :
    - Z > 2.99  : Zone sûre (faible risque de faillite)
    - 1.81 < Z < 2.99 : Zone grise (incertain)
    - Z < 1.81  : Zone de danger (risque élevé)
    """
    if total_assets is None or total_assets == 0:
        return None

    x1 = _safe_divide(working_capital, total_assets, 0.0)
    x2 = _safe_divide(retained_earnings, total_assets, 0.0)
    x3 = _safe_divide(ebit, total_assets, 0.0)
    x4 = _safe_divide(market_cap, total_liabilities, 0.0)
    x5 = _safe_divide(revenue, total_assets, 0.0)

    if None in (x1, x2, x3, x4, x5):
        return None

    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def altman_z_score_label(z: Optional[float]) -> tuple[str, str]:
    """
    Retourne un label texte et une couleur hex pour le Z-Score.
    """
    if z is None:
        return "N/A", "#9ca3af"
    if z > 2.99:
        return "Zone Sûre ✅", "#22c55e"
    elif z > 1.81:
        return "Zone Grise ⚠️", "#f59e0b"
    else:
        return "Zone de Danger 🔴", "#ef4444"


# ─────────────────────────────────────────────
# Piotroski F-Score
# ─────────────────────────────────────────────

def piotroski_f_score(
    # Rentabilité (4 critères)
    net_income: float,
    operating_cash_flow: float,
    roa_current: float,
    roa_previous: float,
    accruals: float,            # OCF/Total assets - ROA
    # Levier / Liquidité (3 critères)
    leverage_current: float,    # LT Debt / Avg Total Assets
    leverage_previous: float,
    current_ratio_current: float,
    current_ratio_previous: float,
    shares_current: float,
    shares_previous: float,
    # Efficacité opérationnelle (3 critères)
    gross_margin_current: float,
    gross_margin_previous: float,
    asset_turnover_current: float,
    asset_turnover_previous: float,
    total_assets: float,
) -> dict:
    """
    Piotroski F-Score : score sur 9 points (9 signaux binaires).

    Interprétation :
    - 8-9 : Entreprise solide (signal d'achat)
    - 4-7 : Neutre
    - 0-3 : Entreprise faible (signal de vente / prudence)

    Retourne un dict avec le score total et le détail de chaque critère.
    """
    scores: dict[str, int] = {}

    # --- Rentabilité ---
    scores["ROA positif"] = 1 if (roa_current is not None and roa_current > 0) else 0
    scores["Cash-flow opérationnel positif"] = 1 if (operating_cash_flow is not None and operating_cash_flow > 0) else 0
    scores["ROA en hausse"] = 1 if (roa_current is not None and roa_previous is not None and roa_current > roa_previous) else 0

    ocf_roa = _safe_divide(operating_cash_flow, total_assets, None) if total_assets else None
    scores["Accruals (qualité des bénéfices)"] = 1 if (ocf_roa is not None and roa_current is not None and ocf_roa > roa_current) else 0

    # --- Levier / Liquidité / Financement ---
    scores["Levier en baisse"] = 1 if (leverage_current is not None and leverage_previous is not None and leverage_current < leverage_previous) else 0
    scores["Liquidité en hausse"] = 1 if (current_ratio_current is not None and current_ratio_previous is not None and current_ratio_current > current_ratio_previous) else 0
    scores["Pas de dilution"] = 1 if (shares_current is not None and shares_previous is not None and shares_current <= shares_previous) else 0

    # --- Efficacité Opérationnelle ---
    scores["Marge brute en hausse"] = 1 if (gross_margin_current is not None and gross_margin_previous is not None and gross_margin_current > gross_margin_previous) else 0
    scores["Rotation actifs en hausse"] = 1 if (asset_turnover_current is not None and asset_turnover_previous is not None and asset_turnover_current > asset_turnover_previous) else 0

    total = sum(scores.values())
    return {"total": total, "details": scores}


def piotroski_label(score: int) -> tuple[str, str]:
    """
    Retourne un label texte et une couleur hex pour le F-Score.
    """
    if score >= 8:
        return "Solide 💪", "#22c55e"
    elif score >= 4:
        return "Neutre 📊", "#f59e0b"
    else:
        return "Faible ⚠️", "#ef4444"


# ─────────────────────────────────────────────
# Scoring Composite (0-100)
# ─────────────────────────────────────────────

def composite_health_score(
    net_margin: Optional[float],
    roe: Optional[float],
    current_ratio_val: Optional[float],
    debt_equity: Optional[float],
    z_score: Optional[float],
    f_score: Optional[int],
) -> Optional[float]:
    """
    Score de santé financière composite normalisé entre 0 et 100.
    Pondération arbitraire mais équilibrée entre rentabilité, liquidité et solvabilité.
    """
    points = 0.0
    total_weight = 0.0

    def add(value: Optional[float], weight: float, low: float, high: float) -> None:
        nonlocal points, total_weight
        if value is None:
            return
        normalized = max(0.0, min(1.0, (value - low) / (high - low)))
        points += normalized * weight
        total_weight += weight

    add(net_margin, 20, -0.1, 0.3)
    add(roe, 20, -0.05, 0.25)
    add(current_ratio_val, 15, 0.5, 3.0)

    # Debt/Equity : plus c'est bas, mieux c'est → on inverse
    if debt_equity is not None:
        inv_de = max(0.0, min(1.0, 1.0 - (debt_equity / 5.0)))
        points += inv_de * 15
        total_weight += 15

    if z_score is not None:
        z_normalized = max(0.0, min(1.0, (z_score - 0) / 5.0))
        points += z_normalized * 15
        total_weight += 15

    if f_score is not None:
        points += (f_score / 9.0) * 15
        total_weight += 15

    if total_weight == 0:
        return None

    return round((points / total_weight) * 100, 1)
