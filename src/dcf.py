"""Forward FCFF DCF.

Layer 1 (this section): pure valuation maths with no company data, so it can be
tested against textbook answers.
Layer 2 (further down): the FCFF forecast engine built on top of it.

Timing convention: flow t (t = 1..N) arrives at the END of year t, or at t-0.5
if mid_year=True. The Gordon terminal value is discounted with the same
convention as the final flow (N, or N-0.5), so mid-year value is exactly
end-year value x (1+WACC)^0.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd


class ValuationError(ValueError):
    """Raised for economically invalid inputs (e.g. terminal growth >= WACC)."""


# --------------------------------------------------------------------------
# Layer 1: textbook building blocks
# --------------------------------------------------------------------------
def check_growth_vs_wacc(wacc: float, g: float) -> None:
    """Guard: a Gordon perpetuity only converges when g < WACC."""
    if wacc <= -1:
        raise ValuationError(f"WACC {wacc} must be > -100%")
    if g >= wacc:
        raise ValuationError(
            f"terminal growth ({g:.4%}) must be strictly below WACC ({wacc:.4%})"
        )


def discount_factors(wacc: float, n: int, mid_year: bool = False) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=float) - (0.5 if mid_year else 0.0)
    return (1.0 + wacc) ** (-t)


def present_value(flows: Sequence[float], wacc: float, mid_year: bool = False) -> float:
    flows = np.asarray(flows, dtype=float)
    return float(np.sum(flows * discount_factors(wacc, len(flows), mid_year)))


def gordon_terminal_value(fcff_next: float, wacc: float, g: float) -> float:
    """Value at end of the explicit period of flows FCFF_{N+1} growing at g forever."""
    check_growth_vs_wacc(wacc, g)
    return fcff_next / (wacc - g)


def enterprise_value(flows: Sequence[float], wacc: float, g: float,
                     mid_year: bool = False) -> float:
    """EV of explicit flows plus Gordon TV. FCFF_{N+1} = FCFF_N x (1+g) (simple textbook form)."""
    flows = np.asarray(flows, dtype=float)
    tv = gordon_terminal_value(flows[-1] * (1.0 + g), wacc, g)
    n = len(flows)
    df_n = float(discount_factors(wacc, n, mid_year)[-1])
    return present_value(flows, wacc, mid_year) + tv * df_n


# --------------------------------------------------------------------------
# Layer 2: FCFF forecast engine
# --------------------------------------------------------------------------
# Definitions (documented in notes/assumptions.md A9-A16):
#   EBIT   = revenue - (total_expenses - finance_costs): operating, AFTER depreciation and AFTER
#            stock-based compensation (SBC is inside employee cost, never added back), BEFORE
#            other income (treasury income belongs to the cash in the bridge) and BEFORE
#            finance costs (incl. Ind AS 116 lease interest -> leases are treated as debt).
#   Tax    = tax_rate x max(0, EBIT - NOL used); losses accumulate in an NOL pool and shelter
#            later profits explicitly. Any NOL left after year N is valued as a residual shield.
#   NOPAT  = EBIT - cash tax.
#   Reinvestment_t = (Revenue_t - Revenue_{t-1}) / sales_to_capital.
#   FCFF   = NOPAT - Reinvestment.
#   Equity = EV + counted cash/investments - debt - leases - NCI ; per share on DILUTED shares.
@dataclass(frozen=True)
class DCFInputs:
    base_revenue: float                 # last reported FY (or LTM) revenue from operations
    base_ebit_margin: float             # operating EBIT margin of that base period (after SBC)
    revenue_cagr: float                 # growth parameter (see growth_profile)
    terminal_growth: float
    terminal_ebit_margin: float
    wacc: float
    sales_to_capital: float
    tax_rate: float                     # statutory rate applied to taxable EBIT in years 1..N
    tax_rate_terminal: float            # rate in the terminal year onward (converges here)
    opening_nol: float = 0.0            # unabsorbed tax losses at base date
    years: int = 10
    margin_ramp_years: int | None = None    # years to reach terminal margin (default: years)
    growth_profile: str = "constant"        # "constant" | "linear_fade"
    mid_year: bool = False
    bridge: dict = field(default_factory=dict)  # name -> signed amount (+asset, -claim)
    diluted_shares: float = 1.0             # same scale as amounts (crore <-> INR crore)

    def with_(self, **kw) -> "DCFInputs":
        return replace(self, **kw)


@dataclass
class DCFResult:
    inputs: DCFInputs
    forecast: pd.DataFrame
    pv_explicit: float
    terminal_fcff: float
    terminal_value: float
    pv_terminal: float
    pv_residual_nol: float
    enterprise_value: float
    bridge: dict
    equity_value: float
    value_per_share: float
    realised_cagr: float
    terminal_marginal_roic: float

    @property
    def tv_share_of_ev(self) -> float:
        return self.pv_terminal / self.enterprise_value if self.enterprise_value else float("nan")


def growth_path(cagr: float, terminal_growth: float, years: int, profile: str) -> np.ndarray:
    """Revenue growth for years 1..N.
    constant:    every year = cagr (so realised CAGR == cagr exactly).
    linear_fade: year 1 = cagr, falling linearly to terminal_growth in year N."""
    if profile == "constant":
        return np.full(years, cagr, dtype=float)
    if profile == "linear_fade":
        if years == 1:
            return np.array([cagr])
        return cagr + (terminal_growth - cagr) * np.arange(years) / (years - 1)
    raise ValuationError(f"unknown growth_profile '{profile}'")


def margin_path(current: float, terminal: float, years: int, ramp_years: int | None) -> np.ndarray:
    """EBIT margin for years 1..N: linear from `current` (year 0) to `terminal`, reached in
    year `ramp_years` and held after."""
    ramp = ramp_years or years
    if not 1 <= ramp <= years:
        raise ValuationError("margin_ramp_years must be between 1 and years")
    k = np.minimum(np.arange(1, years + 1), ramp) / ramp
    return current + (terminal - current) * k


def tax_schedule(ebit: np.ndarray, tax_rate: float, opening_nol: float):
    """Explicit NOL roll-forward. Returns (tax, nol_open, nol_used, nol_added, nol_close)."""
    n = len(ebit)
    tax = np.zeros(n); o = np.zeros(n); used = np.zeros(n); added = np.zeros(n); c = np.zeros(n)
    bal = opening_nol
    for i, e in enumerate(ebit):
        o[i] = bal
        if e > 0:
            used[i] = min(bal, e)
            tax[i] = tax_rate * (e - used[i])
        else:
            added[i] = -e
        bal = bal - used[i] + added[i]
        c[i] = bal
    return tax, o, used, added, c


def run_dcf(inp: DCFInputs) -> DCFResult:
    n = inp.years
    check_growth_vs_wacc(inp.wacc, inp.terminal_growth)
    if inp.sales_to_capital <= 0:
        raise ValuationError("sales_to_capital must be > 0")
    if inp.diluted_shares <= 0:
        raise ValuationError("diluted_shares must be > 0")

    g = growth_path(inp.revenue_cagr, inp.terminal_growth, n, inp.growth_profile)
    revenue = inp.base_revenue * np.cumprod(1.0 + g)
    prev = np.r_[inp.base_revenue, revenue[:-1]]
    margin = margin_path(inp.base_ebit_margin, inp.terminal_ebit_margin, n, inp.margin_ramp_years)
    ebit = revenue * margin
    tax, nol_o, nol_u, nol_a, nol_c = tax_schedule(ebit, inp.tax_rate, inp.opening_nol)
    nopat = ebit - tax
    reinvest = (revenue - prev) / inp.sales_to_capital
    fcff = nopat - reinvest
    df = discount_factors(inp.wacc, n, inp.mid_year)

    # Terminal year (N+1): margin at target, terminal tax rate, no NOL left assumed to matter,
    # reinvestment consistent with sales-to-capital at the terminal growth rate.
    rev_t = revenue[-1] * (1.0 + inp.terminal_growth)
    ebit_t = rev_t * inp.terminal_ebit_margin
    nopat_t = ebit_t * (1.0 - inp.tax_rate_terminal) if ebit_t > 0 else ebit_t
    reinv_t = (rev_t - revenue[-1]) / inp.sales_to_capital
    fcff_t = nopat_t - reinv_t
    tv = gordon_terminal_value(fcff_t, inp.wacc, inp.terminal_growth)
    pv_tv = tv * df[-1]

    # NOL still unused after year N: value at the tax shield, lump-sum discounted from year N
    # (conservative: it would in reality be used gradually after N).
    pv_nol = inp.tax_rate_terminal * nol_c[-1] * df[-1]

    pv_explicit = float(np.sum(fcff * df))
    ev = pv_explicit + pv_tv + pv_nol
    equity = ev + sum(inp.bridge.values())
    marginal_roic = inp.sales_to_capital * inp.terminal_ebit_margin * (1.0 - inp.tax_rate_terminal)

    table = pd.DataFrame({
        "year": np.arange(0, n + 1),
        "revenue": np.r_[inp.base_revenue, revenue],
        "growth": np.r_[np.nan, g],
        "ebit_margin": np.r_[inp.base_ebit_margin, margin],
        "ebit": np.r_[inp.base_ebit_margin * inp.base_revenue, ebit],
        "nol_open": np.r_[np.nan, nol_o],
        "nol_used": np.r_[np.nan, nol_u],
        "nol_added": np.r_[np.nan, nol_a],
        "nol_close": np.r_[inp.opening_nol, nol_c],
        "cash_tax": np.r_[np.nan, tax],
        "nopat": np.r_[np.nan, nopat],
        "reinvestment": np.r_[np.nan, reinvest],
        "fcff": np.r_[np.nan, fcff],
        "discount_factor": np.r_[np.nan, df],
        "pv_fcff": np.r_[np.nan, fcff * df],
    })
    return DCFResult(
        inputs=inp, forecast=table, pv_explicit=pv_explicit, terminal_fcff=fcff_t,
        terminal_value=tv, pv_terminal=pv_tv, pv_residual_nol=pv_nol, enterprise_value=ev,
        bridge=dict(inp.bridge), equity_value=equity,
        value_per_share=equity / inp.diluted_shares,
        realised_cagr=(revenue[-1] / inp.base_revenue) ** (1.0 / n) - 1.0,
        terminal_marginal_roic=marginal_roic,
    )


def value_per_share(inp: DCFInputs) -> float:
    """Thin wrapper used by the reverse solver (M3)."""
    return run_dcf(inp).value_per_share


# --------------------------------------------------------------------------
# Equity bridge and base metrics from loaded data
# --------------------------------------------------------------------------
def operating_ebit(total_revenue: float, total_expenses: float, finance_costs: float) -> float:
    """EBIT = revenue - (total_expenses - finance_costs). Excludes other income and finance costs."""
    return total_revenue - (total_expenses - finance_costs)


def build_bridge(values: dict, weights: dict, claims: Sequence[str]) -> dict:
    """Signed bridge lines. `weights` maps ASSET items to the fraction counted (1 = fully liquid,
    0 = excluded); every item in `weights` is returned (even at 0) so the workbook can show what
    was counted. `claims` are deducted at book value. Missing values raise, never default to 0."""
    lines = {}
    for item, wgt in weights.items():
        if item not in values:
            raise ValuationError(f"bridge asset item '{item}' has no value")
        lines[item] = float(wgt) * float(values[item])
    for item in claims:
        if item not in values:
            raise ValuationError(f"bridge claim item '{item}' has no value")
        lines[item] = -float(values[item])
    return lines
