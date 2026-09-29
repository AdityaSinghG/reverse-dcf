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
