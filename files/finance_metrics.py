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
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return default
    if numerator is None or pd.isna(numerator):
        return default
    
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return default


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


# ─────────────────────────────────────────────
# Monte Carlo Simulation
# ─────────────────────────────────────────────

import numpy as np


def monte_carlo_simulation(
    last_price: float,
    daily_returns: "pd.Series | None" = None,
    mu: float | None = None,
    sigma: float | None = None,
    days: int = 252,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Simule n trajectoires de prix via mouvement brownien géométrique.

    Paramètres :
    - last_price : dernier cours connu
    - daily_returns : Series de rendements quotidiens historiques (optionnel)
    - mu, sigma : moyenne et écart-type quotidiens (calculés depuis daily_returns si None)
    - days : horizon en jours de trading (défaut 252 = 1 an)
    - n_simulations : nombre de trajectoires (défaut 1000)

    Retourne un dict avec :
    - simulations : array (n_simulations, days) de prix simulés
    - stats : dict avec mean, median, p5, p95, last_price
    """
    rng = np.random.default_rng(seed)

    if daily_returns is not None and len(daily_returns) > 10:
        dr = daily_returns.dropna()
        if mu is None:
            mu = float(dr.mean())
        if sigma is None:
            sigma = float(dr.std())
    else:
        # Fallback : 8% annuel, 25% vol
        if mu is None:
            mu = 0.08 / 252
        if sigma is None:
            sigma = 0.25 / np.sqrt(252)

    # GBM : S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    dt = 1.0
    drift = (mu - 0.5 * sigma ** 2) * dt
    shock = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_simulations, days))
    daily_growth = np.exp(drift + shock * Z)

    price_paths = np.zeros((n_simulations, days + 1))
    price_paths[:, 0] = last_price

    for t in range(days):
        price_paths[:, t + 1] = price_paths[:, t] * daily_growth[:, t]

    final_prices = price_paths[:, -1]

    return {
        "simulations": price_paths,
        "stats": {
            "last_price": last_price,
            "mean": float(np.mean(final_prices)),
            "median": float(np.median(final_prices)),
            "p5": float(np.percentile(final_prices, 5)),
            "p95": float(np.percentile(final_prices, 95)),
            "std": float(np.std(final_prices)),
        },
    }


# ─────────────────────────────────────────────
# DCF (Discounted Cash Flow) Valuation
# ─────────────────────────────────────────────

def dcf_valuation(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    terminal_growth: float = 0.025,
    discount_rate: float = 0.10,
    projection_years: int = 5,
    shares_outstanding: float | None = None,
) -> dict:
    """
    Modèle DCF simplifié.

    Paramètres :
    - free_cash_flow : dernier Free Cash Flow annuel
    - growth_rate : taux de croissance projeté (ex: 0.05 = 5%)
    - terminal_growth : croissance perpétuelle (ex: 0.025 = 2.5%)
    - discount_rate : WACC ou coût du capital (ex: 0.10 = 10%)
    - projection_years : nombre d'années de projection
    - shares_outstanding : nombre d'actions (pour prix par action)

    Retourne un dict avec les projections, la valeur terminale, et la fair value.
    """
    if discount_rate <= terminal_growth:
        return {"error": "Le taux d'actualisation doit être supérieur à la croissance perpétuelle."}

    projected_fcfs = []
    pv_fcfs = []
    fcf = free_cash_flow

    for yr in range(1, projection_years + 1):
        fcf = fcf * (1 + growth_rate)
        pv = fcf / ((1 + discount_rate) ** yr)
        projected_fcfs.append({"year": yr, "fcf": fcf, "pv": pv})
        pv_fcfs.append(pv)

    # Terminal value (Gordon Growth Model)
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)

    enterprise_value = sum(pv_fcfs) + pv_terminal

    result = {
        "projections": projected_fcfs,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "sum_pv_fcfs": sum(pv_fcfs),
    }

    if shares_outstanding and shares_outstanding > 0:
        result["fair_value_per_share"] = enterprise_value / shares_outstanding
    else:
        result["fair_value_per_share"] = None

    return result
