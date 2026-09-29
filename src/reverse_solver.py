"""Reverse DCF: which single input makes the forward DCF equal the market price?

For one parameter at a time (revenue CAGR, terminal EBIT margin, WACC) we solve
    f(x) = value_per_share(inputs with parameter = x) - price = 0
holding every other input fixed.

Why a scan before brentq: value is NOT guaranteed monotone in every input. With the loss-pool tax
logic and reinvestment, extra growth can destroy value (marginal ROIC < WACC) or first deepen losses
then repay them, so f can have zero, one or several roots. brentq needs a sign change and returns
only one root, so we first scan the bracket on a fixed grid to find every sign change, then run
brentq inside each one. Outcomes are explicit:
    SOLVED                 exactly one root in the bracket
    MULTIPLE_ROOTS         >1 root found: all are listed, none is picked for you
    NO_SOLUTION_IN_RANGE   no sign change in the bracket. The bracket is NEVER widened silently.
A root sitting between two adjacent grid points that both overshoot (a double root touching the
price) can be missed; the grid is dense (default 400 points) to make that unlikely.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src import dcf

PARAMETERS = ("revenue_cagr", "terminal_ebit_margin", "wacc")

# Explicit default brackets (annual rates / margins). Override per call; they are printed in output.
DEFAULT_BRACKETS = {
    "revenue_cagr": (-0.05, 0.60),
    "terminal_ebit_margin": (-0.05, 0.60),
    "wacc": (0.05, 0.30),
}

SOLVED, MULTIPLE, NO_SOLUTION = "SOLVED", "MULTIPLE_ROOTS", "NO_SOLUTION_IN_RANGE"


@dataclass
class SolveResult:
    parameter: str
    status: str
    value: float | None                 # the root when status == SOLVED, else None
    roots: list[float]
    target_price: float
    bracket_requested: tuple[float, float]
    bracket_used: tuple[float, float]   # differs from requested only where a guard forces it
    value_range_in_bracket: tuple[float, float]   # min / max forward value seen on the scan
    n_invalid_points: int               # grid points where inputs were invalid (e.g. g >= WACC)
    held_fixed: dict
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"IMPLIED {self.parameter}  (target price {self.target_price:,.4f})",
                 f"  status  : {self.status}"]
        if self.status == SOLVED:
            lines.append(f"  implied : {self.value:.6f}")
        elif self.status == MULTIPLE:
            lines.append("  roots   : " + ", ".join(f"{r:.6f}" for r in self.roots))
        else:
            lo, hi = self.value_range_in_bracket
            lines.append(f"  no solution in range: model value spans {lo:,.2f} .. {hi:,.2f} "
                         f"over the bracket, price is outside it")
        lines.append(f"  bracket : {self.bracket_used}"
                     + ("" if self.bracket_used == self.bracket_requested
                        else f"  (requested {self.bracket_requested})"))
        for n in self.notes:
            lines.append(f"  note    : {n}")
        lines.append("  held fixed:")
        for k, v in self.held_fixed.items():
            lines.append(f"    {k:<22} {v}")
        return "\n".join(lines)

    __str__ = report


def held_fixed(inp: dcf.DCFInputs, exclude: tuple[str, ...] = ()) -> dict:
    """Every input that is NOT being solved for, in printable form."""
    out = {}
    for f in fields(inp):
        if f.name in exclude:
            continue
        v = getattr(inp, f.name)
        if f.name == "bridge":
            out["bridge_net_claims"] = round(sum(v.values()), 4)
            out["bridge_lines"] = {k: round(x, 4) for k, x in v.items()}
        else:
            out[f.name] = v
    return out


def _scan(f: Callable[[float], float], lo: float, hi: float, n: int):
    xs = np.linspace(lo, hi, n)
    ys = np.full(n, np.nan)
    for i, x in enumerate(xs):
        try:
            ys[i] = f(x)
        except dcf.ValuationError:
            pass
    return xs, ys


def solve_implied(inp: dcf.DCFInputs, price: float, parameter: str,
                  bracket: tuple[float, float] | None = None, n_scan: int = 400) -> SolveResult:
    """Solve for `parameter` so that forward value per share == price; others held at `inp`."""
    if parameter not in PARAMETERS:
        raise ValueError(f"parameter must be one of {PARAMETERS}")
    if price <= 0:
        raise ValueError("price must be > 0")
    requested = tuple(bracket) if bracket else DEFAULT_BRACKETS[parameter]
    lo, hi = requested
    notes: list[str] = []
    if not lo < hi:
        raise ValueError("bracket must have lo < hi")
    if parameter == "wacc":
        # Guard, not a widening: WACC must exceed terminal growth. Raise the floor and say so.
        floor = inp.terminal_growth + 1e-4
        if lo < floor:
            notes.append(f"WACC lower bound raised from {lo} to {floor:.4f} (must exceed terminal growth)")
            lo = floor
        if lo >= hi:
            raise ValueError("bracket is entirely below terminal growth")

    def f(x: float) -> float:
        return dcf.value_per_share(inp.with_(**{parameter: x})) - price

    xs, ys = _scan(f, lo, hi, n_scan)
    valid = ~np.isnan(ys)
    n_invalid = int((~valid).sum())
    if n_invalid:
        notes.append(f"{n_invalid} grid points skipped as invalid inputs (e.g. terminal growth >= WACC)")
    fx = ys[valid] + price
    value_range = (float(fx.min()), float(fx.max())) if valid.any() else (float("nan"),) * 2

    roots: list[float] = []
    vx, vy = xs[valid], ys[valid]
    for i in range(len(vx)):
        if vy[i] == 0.0:
            roots.append(float(vx[i]))
        if i + 1 < len(vx) and vy[i] * vy[i + 1] < 0:
            r = brentq(f, vx[i], vx[i + 1], xtol=1e-13, rtol=1e-13, maxiter=500)
            roots.append(float(r))
    roots = sorted(set(round(r, 12) for r in roots))

    status = NO_SOLUTION if not roots else (SOLVED if len(roots) == 1 else MULTIPLE)
    if status == MULTIPLE:
        notes.append("value is non-monotone in this parameter (loss pool / growth destroying value); "
                     "choose the economically sensible root yourself")
    return SolveResult(
        parameter=parameter, status=status, value=roots[0] if status == SOLVED else None,
        roots=roots, target_price=price, bracket_requested=requested, bracket_used=(lo, hi),
        value_range_in_bracket=value_range, n_invalid_points=n_invalid,
        held_fixed=held_fixed(inp, exclude=(parameter,)), notes=notes)


def solve_all(inp: dcf.DCFInputs, price: float, brackets: dict | None = None) -> dict[str, SolveResult]:
    """The three one-at-a-time solves, each with its own held-fixed report."""
    brackets = brackets or {}
    return {p: solve_implied(inp, price, p, brackets.get(p)) for p in PARAMETERS}


def implied_frontier(inp: dcf.DCFInputs, price: float, cagrs, margin_bracket=None,
                     n_scan: int = 400) -> pd.DataFrame:
    """2D frontier: for each revenue CAGR, the terminal EBIT margin that makes value == price.
    Every point on the curve satisfies value_per_share == price (WACC, g, S2C, tax fixed).
    Points with no root or several roots are kept with status so gaps are visible, not hidden."""
    rows = []
    for c in cagrs:
        r = solve_implied(inp.with_(revenue_cagr=float(c)), price, "terminal_ebit_margin",
                          margin_bracket, n_scan)
        rows.append({"revenue_cagr": float(c), "implied_terminal_margin": r.value,
                     "status": r.status, "all_roots": r.roots})
    out = pd.DataFrame(rows)
    out.attrs["held_fixed"] = held_fixed(inp, exclude=("revenue_cagr", "terminal_ebit_margin"))
    out.attrs["price"] = price
    return out
