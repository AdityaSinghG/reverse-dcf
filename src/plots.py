"""matplotlib charts, saved as PNG under plots/.

Every chart carries a title and a source note (the `source` argument is required, so no chart can be
produced without saying where its numbers came from). Synthetic runs must pass a source that starts
with "TEST".

Style follows the dataviz method: one accent per job (model/implied = blue, market price = orange),
neutral grey for reference data, thin marks, quiet grid, no dual axes, direct labels, text in ink
colours never in series colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, NEUTRAL, BAND = "#2a78d6", "#eb6834", "#a9a8a1", "#dcdad2"


def _base(title: str, subtitle: str | None = None, size=(9.5, 5.6)):
    fig, ax = plt.subplots(figsize=size, dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.06, 0.918, subtitle, ha="left", fontsize=9.5, color=INK2)
    return fig, ax


def _finish(fig, source: str, path, top=0.86, bottom=0.14, left=0.11):
    if not source or not source.strip():
        raise ValueError("every chart needs a source note")
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=0.96)
    fig.text(0.06, 0.02, "Source: " + source, ha="left", va="bottom", fontsize=7.5,
             color=INK2, wrap=True)
    path = Path(path)
    if not path.is_absolute():
        path = PLOTS_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def cagr(first: float, last: float, years: float) -> float:
    """Compound annual growth rate between two values `years` apart."""
    if first <= 0 or last <= 0 or years <= 0:
        raise ValueError("CAGR needs positive endpoints and years > 0")
    return (last / first) ** (1.0 / years) - 1.0


# 1 ---------------------------------------------------------------------------------------------------
def plot_implied_vs_historical_cagr(implied: float, historical: dict[str, float], *, title: str,
                                    source: str, path="implied_vs_historical_cagr.png",
                                    subtitle: str | None = None):
    """Horizontal bars: the CAGR the price implies (blue) against historical / peer CAGRs (grey)."""
    items = dict(historical)
    items["Implied by market price"] = implied
    labels = sorted(items, key=lambda k: items[k])
    vals = [items[k] for k in labels]
    fig, ax = _base(title, subtitle, size=(9.5, max(5.0, 1.1 + 0.55 * len(labels) + 1.8)))
    colors = [BLUE if k == "Implied by market price" else NEUTRAL for k in labels]
    bars = ax.barh(labels, vals, color=colors, height=0.55, edgecolor=SURFACE, linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(v + (0.004 if v >= 0 else -0.004), b.get_y() + b.get_height() / 2, f"{v:.1%}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Revenue CAGR", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, min(vals)) - 0.02, max(vals) * 1.15 + 0.02)
    return _finish(fig, source, path, top=0.84, bottom=0.16, left=0.27)


# 2 ---------------------------------------------------------------------------------------------------
def plot_frontier(frontier: pd.DataFrame, *, title: str, source: str,
                  margin_band: tuple[float, float, str] | None = None,
                  cagr_band: tuple[float, float, str] | None = None,
                  points: dict[str, tuple[float, float]] | None = None,
                  path="implied_frontier.png", subtitle: str | None = None):
    """CAGR (x) vs terminal EBIT margin (y) combinations that reproduce the market price.
    Bands shade the historical / peer ranges. Only SOLVED points are drawn; gaps stay gaps."""
    ok = frontier[frontier["status"] == "SOLVED"]
    fig, ax = _base(title, subtitle)
    if margin_band:
        lo, hi, lab = margin_band
        ax.axhspan(lo, hi, color=BAND, alpha=0.7, lw=0)
        ax.text(ok["revenue_cagr"].max(), hi, lab + "  ", va="bottom", ha="right",
                fontsize=8.5, color=INK2)
    if cagr_band:
        lo, hi, lab = cagr_band
        ax.axvspan(lo, hi, color=BAND, alpha=0.45, lw=0)
        ax.text(hi, ok["implied_terminal_margin"].max(), " " + lab, va="top", ha="left",
                fontsize=8.5, color=INK2)
    ax.plot(ok["revenue_cagr"], ok["implied_terminal_margin"], color=BLUE, lw=2, zorder=3)
    if len(ok) != len(frontier):
        ax.text(0.99, 0.02, f"{len(frontier) - len(ok)} grid points had no unique solution (omitted)",
                transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    for name, (x, y) in (points or {}).items():
        ax.scatter([x], [y], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(name, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=8.5, color=INK)
    ax.text(ok["revenue_cagr"].iloc[len(ok) // 2], ok["implied_terminal_margin"].iloc[len(ok) // 2],
            "  combinations that equal the market price", color=INK, fontsize=9, va="bottom")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Revenue CAGR, years 1-10", color=INK2, fontsize=9)
    ax.set_ylabel("Terminal EBIT margin", color=INK2, fontsize=9)
    return _finish(fig, source, path)


# 3 ---------------------------------------------------------------------------------------------------
def plot_mc_histogram(result, *, title: str, source: str, uncorrelated=None,
                      path="mc_histogram.png", subtitle: str | None = None, bins: int = 60):
    """Histogram of simulated value per share with the market price line and P(value >= price).
    `result` is the (correlated) run; pass `uncorrelated` to annotate both probabilities."""
    v = result.values
    lo, hi = np.percentile(v, [0.5, 99.5])
    fig, ax = _base(title, subtitle)
    ax.hist(v, bins=bins, range=(lo, hi), color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    n_out = int(((v < lo) | (v > hi)).sum())
    ax.axvline(result.price, color=ORANGE, lw=2, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.text(result.price, ymax * 0.98, f" Market price {result.price:,.0f}", color=INK, fontsize=9,
            va="top", ha="left" if result.price < (lo + hi) / 2 else "right")
    lines = [f"P(value >= price) = {result.prob_ge_price:.1%}"
             f"   (corr {result.corr_cagr_margin:+.1f}, ±{result.prob_se:.1%} s.e.)"]
    if uncorrelated is not None:
        lines.append(f"P(value >= price) = {uncorrelated.prob_ge_price:.1%}   (uncorrelated)")
    lines.append(f"median {result.stats['p50']:,.0f}  |  5th-95th pct {result.stats['p5']:,.0f} to "
                 f"{result.stats['p95']:,.0f}")
    if n_out:
        lines.append(f"axis trimmed to 0.5th-99.5th pct ({n_out} draws outside not shown; stats use all)")
    lines.append(f"n = {result.n_sims:,}, seed {result.seed}, g>=WACC redrawn: "
                 f"{result.n_rejected_g_ge_wacc:,}, DCF errors: {result.n_errors}")
    ax.text(0.98, 0.95 if result.price < (lo + hi) / 2 else 0.95, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color=INK, linespacing=1.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Value per share (currency of the model)", color=INK2, fontsize=9)
    ax.set_ylabel("Simulations", color=INK2, fontsize=9)
    ax.grid(axis="x", visible=False)
    return _finish(fig, source, path)


# 4 ---------------------------------------------------------------------------------------------------
def plot_tornado(table: pd.DataFrame, *, title: str, source: str, price: float | None = None,
                 path="tornado.png", subtitle: str | None = None, labels: dict | None = None):
    """Value per share when each driver moves to its P10 (orange) / P90 (blue), others at base."""
    t = table.dropna(subset=["swing"]).iloc[::-1]          # largest swing on top
    base_v = table.attrs["base_value"]
    pretty = {"revenue_cagr": "Revenue CAGR", "terminal_ebit_margin": "Terminal EBIT margin",
              "wacc": "WACC", "terminal_growth": "Terminal growth", **(labels or {})}
    names = [pretty.get(d, d) for d in t["driver"]]
    fig, ax = _base(title, subtitle, size=(9.5, 1.6 + 0.7 * len(t) + 1.6))
    y = np.arange(len(t))
    for i, r in enumerate(t.itertuples()):
        for val, col in ((r.value_at_low, ORANGE), (r.value_at_high, BLUE)):
            ax.barh(i, val - base_v, left=base_v, height=0.5, color=col, edgecolor=SURFACE,
                    linewidth=1.5, zorder=3)
        pct = lambda x: f"{x:.1%}"
        ax.text(min(r.value_at_low, r.value_at_high) - 0.01 * abs(base_v), i,
                pct(r.input_low if r.value_at_low < r.value_at_high else r.input_high) + "  ",
                ha="right", va="center", fontsize=8.3, color=INK)
        ax.text(max(r.value_at_low, r.value_at_high) + 0.01 * abs(base_v), i,
                "  " + pct(r.input_high if r.value_at_low < r.value_at_high else r.input_low),
                ha="left", va="center", fontsize=8.3, color=INK)
    ax.set_yticks(y, names)
    ax.axvline(base_v, color=INK2, lw=1, zorder=4)
    ax.set_ylim(-0.6, len(t) - 0.1 + 0.2)
    lo_x = min(t["value_at_low"].min(), t["value_at_high"].min())
    hi_x = max(t["value_at_low"].max(), t["value_at_high"].max())
    pad = 0.12 * (hi_x - lo_x)
    ax.set_xlim(min(lo_x, price if price is not None else lo_x) - pad, hi_x + pad)
    ax.text(base_v, len(t) - 0.15 + 0.15, f"base {base_v:,.0f}", ha="left" if (price is None or price < base_v) else "right",
            va="top", fontsize=8.5, color=INK2)
    if price is not None:
        ax.axvline(price, color=NEUTRAL, lw=1.4, ls="--", zorder=2)
        ax.text(price, len(t) - 0.15 + 0.15, f"price {price:,.0f} ", ha="right" if price < base_v else "left",
                va="top", fontsize=8.5, color=INK2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="driver at P10 of its assumed range"),
                       Patch(color=BLUE, label="driver at P90")], loc="lower right", frameon=False,
              fontsize=8.5, labelcolor=INK2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Value per share", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    return _finish(fig, source, path, top=0.85, bottom=0.17, left=0.22)
