"""Cost of capital: CAPM cost of equity, regression beta, peer sanity check, WACC.

Conventions (see notes/assumptions.md A6, A12):
  * Everything is nominal in the cash-flow currency (INR for Eternal): rf is the INR
    10Y G-Sec yield and the ERP is Damodaran's India total ERP (mature-market ERP + country
    risk premium). Never mix a USD rf with INR cash flows.
  * Beta is an OLS slope of stock simple returns on index simple returns:
        r_stock,t = alpha + beta * r_index,t + e_t
    on a stated window and frequency, with standard error and R^2 reported so the
    imprecision is visible. A Blume adjustment (2/3*raw + 1/3) is available, off by default.
  * Debt for weights and for unlevering INCLUDES lease liabilities, matching the DCF,
    which deducts leases as debt in the equity bridge (Ind AS 116 EBIT is pre-lease-interest).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class WaccError(ValueError):
    pass


# --------------------------------------------------------------------------
# CAPM / WACC arithmetic
# --------------------------------------------------------------------------
def cost_of_equity(rf: float, beta: float, erp: float) -> float:
    """CAPM: Ke = rf + beta x ERP."""
    return rf + beta * erp


def after_tax_cost_of_debt(kd_pre_tax: float, tax_rate: float) -> float:
    """Kd x (1 - t). Uses the marginal statutory rate: interest is deducted at the margin."""
    return kd_pre_tax * (1.0 - tax_rate)


def capital_weights(equity_value: float, debt_value: float) -> tuple[float, float]:
    """(E/V, D/V) from market equity and (book proxy) debt incl. leases."""
    if equity_value <= 0 or debt_value < 0:
        raise WaccError("equity value must be > 0 and debt value >= 0")
    v = equity_value + debt_value
    return equity_value / v, debt_value / v


def wacc(ke: float, kd_after_tax: float, w_equity: float, w_debt: float) -> float:
    if abs(w_equity + w_debt - 1.0) > 1e-9:
        raise WaccError(f"weights must sum to 1, got {w_equity + w_debt}")
    return w_equity * ke + w_debt * kd_after_tax


def blume_adjust(beta_raw: float) -> float:
    """Blume (1971) mean-reversion: 2/3 raw + 1/3 x 1.0."""
    return (2.0 / 3.0) * beta_raw + 1.0 / 3.0


def unlever_beta(beta_levered: float, debt: float, equity: float, tax_rate: float) -> float:
    """Hamada: Bu = BL / (1 + (1-t) D/E)."""
    return beta_levered / (1.0 + (1.0 - tax_rate) * debt / equity)


def relever_beta(beta_unlevered: float, debt: float, equity: float, tax_rate: float) -> float:
    return beta_unlevered * (1.0 + (1.0 - tax_rate) * debt / equity)


# --------------------------------------------------------------------------
# Price data (provenance enforced via header comments)
# --------------------------------------------------------------------------
def load_price_series(path: str | Path) -> pd.Series:
    """CSV with header comment lines `# source: ...` and `# retrieved: YYYY-MM-DD`,
    then `date,close`. Returns close indexed by date. Refuses files without provenance."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    header = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            k, _, v = line[1:].partition(":")
            header[k.strip().lower()] = v.strip()
    for key in ("source", "retrieved"):
        if not header.get(key):
            raise WaccError(
                f"{path.name}: missing '# {key}: ...' header (rule: no data without provenance)")
    df = pd.read_csv(path, comment="#", parse_dates=["date"])
    s = df.set_index("date")["close"].astype(float).sort_index()
    if s.index.has_duplicates or (s <= 0).any():
        raise WaccError(f"{path.name}: duplicate dates or non-positive prices")
    s.attrs.update(header)
    return s


def periodic_returns(prices: pd.Series, freq: str) -> pd.Series:
    """Simple returns on last-close-of-period prices. freq: 'W' (week ending Friday) or 'M'."""
    rule = {"W": "W-FRI", "M": "ME"}.get(freq)
    if rule is None:
        raise WaccError("freq must be 'W' or 'M'")
    return prices.resample(rule).last().dropna().pct_change().dropna()


# --------------------------------------------------------------------------
# Regression beta
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BetaResult:
    beta: float
    std_err: float
    r_squared: float
    alpha: float
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str
    window_years: float | None
    index_name: str

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.beta - 1.96 * self.std_err, self.beta + 1.96 * self.std_err)

    @property
    def blume(self) -> float:
        return blume_adjust(self.beta)


def regression_beta(stock: pd.Series, index: pd.Series, *, freq: str = "W",
                    window_years: float | None = 2.0, end: str | pd.Timestamp | None = None,
                    index_name: str = "NIFTY 50") -> BetaResult:
    """OLS beta of stock on index returns over the trailing `window_years` (None = full history)."""
    r = pd.concat({"s": periodic_returns(stock, freq), "m": periodic_returns(index, freq)},
                  axis=1).dropna()
    end_ts = pd.Timestamp(end) if end is not None else r.index.max()
    r = r[r.index <= end_ts]
    if window_years is not None:
        r = r[r.index > end_ts - pd.DateOffset(days=int(round(365.25 * window_years)))]
    n = len(r)
    if n < 20:
        raise WaccError(f"only {n} return observations: too few for a beta regression")
    x, y = r["m"].to_numpy(), r["s"].to_numpy()
    xm = x - x.mean()
    beta = float(xm @ (y - y.mean()) / (xm @ xm))
    alpha = float(y.mean() - beta * x.mean())
    resid = y - alpha - beta * x
    se = float(np.sqrt(resid @ resid / (n - 2) / (xm @ xm)))
    r2 = float(1.0 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())))
    return BetaResult(beta, se, r2, alpha, n, r.index.min(), r.index.max(), freq,
                      window_years, index_name)


def beta_window_grid(stock: pd.Series, index: pd.Series, *, end=None,
                     windows=(2.0, 3.0, 5.0, None), freqs=("W", "M"),
                     index_name: str = "NIFTY 50") -> pd.DataFrame:
    """Beta under alternative windows/frequencies, so the chosen method is visibly one of many."""
    rows = []
    for win in windows:
        for f in freqs:
            label = win if win is not None else "full"
            try:
                b = regression_beta(stock, index, freq=f, window_years=win, end=end,
                                    index_name=index_name)
                rows.append({"window_years": label, "freq": f, "beta": b.beta,
                             "std_err": b.std_err, "r2": b.r_squared, "n_obs": b.n_obs})
            except WaccError:
                rows.append({"window_years": label, "freq": f, "beta": np.nan,
                             "std_err": np.nan, "r2": np.nan, "n_obs": 0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Peer sanity check
# --------------------------------------------------------------------------
PEER_COLUMNS = ["name", "levered_beta", "debt", "market_cap", "tax_rate", "source", "source_date"]


def load_peer_inputs(path: str | Path) -> pd.DataFrame:
    """Peer table; every row needs source + source_date. Debt should include leases."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    df = pd.read_csv(path)
    miss = [c for c in PEER_COLUMNS if c not in df.columns]
    if miss:
        raise WaccError(f"{path.name}: missing columns {miss}")
    if df[["source", "source_date"]].isna().any().any():
        raise WaccError(f"{path.name}: every peer row needs source and source_date")
    return df


def peer_beta_check(peers: pd.DataFrame, *, target_debt: float, target_equity: float,
                    target_tax: float) -> pd.DataFrame:
    """Unlever each peer, take the median, relever at the target's capital structure.
    Returns peers with `unlevered_beta`, plus a final row for the relevered median."""
    out = peers.copy()
    out["unlevered_beta"] = [
        unlever_beta(r.levered_beta, r.debt, r.market_cap, r.tax_rate) for r in out.itertuples()
    ]
    med = float(out["unlevered_beta"].median())
    summary = {
        "name": "MEDIAN (relevered at target D/E)",
        "unlevered_beta": med,
        "levered_beta": relever_beta(med, target_debt, target_equity, target_tax),
    }
    return pd.concat([out, pd.DataFrame([summary])], ignore_index=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m src.wacc <stock_prices.csv> <index_prices.csv> [end_date]. "
                         "Prints the regression beta (2y weekly) and the window/frequency grid for A6; "
                         "transcribe beta and beta_se into the market data file with the source noted.")
    stock, index = load_price_series(sys.argv[1]), load_price_series(sys.argv[2])
    end = sys.argv[3] if len(sys.argv) > 3 else None
    r = regression_beta(stock, index, freq="W", window_years=2.0, end=end)
    print(f"beta {r.beta:.3f}  s.e. {r.std_err:.3f}  R^2 {r.r_squared:.3f}  n {r.n_obs}  "
          f"{r.start.date()} to {r.end.date()}  ({r.freq}, {r.window_years}y vs {r.index_name})")
    print(beta_window_grid(stock, index, end=end).to_string(index=False))
