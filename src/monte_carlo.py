"""Monte Carlo over the forward DCF.

Four inputs are drawn: revenue CAGR, terminal EBIT margin, WACC, terminal growth. Everything
else (base revenue, sales-to-capital, tax, loss pool, bridge, diluted shares) is held at the
base inputs.

Rules this module enforces:
  * Distribution types/parameters come from config and are ASSUMPTIONS. They are never fitted
    to the market price (that would make P(value >= price) circular).
  * Seeded: same seed + same spec -> identical draws and identical output. Seed is recorded.
  * Correlation between CAGR and terminal margin uses a Gaussian copula, so it works with any
    marginal type. The uncorrelated and correlated runs use the SAME underlying random numbers
    (common random numbers), so differences between them are due to correlation alone.
  * g < WACC is enforced by REJECTION and redraw. The count and share of rejected draws are
    reported, with accepted vs unconditional means of WACC and g, so a biased sample is visible.
  * A draw that raises inside the DCF is counted by error type and excluded from statistics;
    n_valid + n_errors == n_sims always.

Supported marginals (spec dicts): normal{mean,sd}; truncated_normal{mean,sd,low,high};
triangular{low,mode,high}; uniform{low,high}; fixed{value}.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src import dcf

DRAWN = ("revenue_cagr", "terminal_ebit_margin", "wacc", "terminal_growth")
_COLS = {name: i for i, name in enumerate(DRAWN)}


class MonteCarloError(ValueError):
    pass


class _Fixed:
    def __init__(self, v: float):
        self.v = float(v)

    def ppf(self, u):
        return np.full(np.shape(u), self.v)


def make_dist(spec: dict):
    """Frozen distribution with a .ppf (inverse CDF) from a config spec. Nulls raise."""
    t = spec.get("type")
    need = {"normal": ("mean", "sd"), "truncated_normal": ("mean", "sd", "low", "high"),
            "triangular": ("low", "mode", "high"), "uniform": ("low", "high"), "fixed": ("value",)}
    if t not in need:
        raise MonteCarloError(f"unknown distribution type '{t}'; use one of {sorted(need)}")
    missing = [k for k in need[t] if spec.get(k) is None]
    if missing:
        raise MonteCarloError(f"{t} distribution has unset parameters {missing} "
                              "(these are ASSUMPTIONS you must set; see notes/assumptions.md)")
    p = {k: float(spec[k]) for k in need[t]}
    if t == "normal":
        if p["sd"] <= 0:
            raise MonteCarloError("sd must be > 0 (use type 'fixed' for zero variance)")
        return stats.norm(p["mean"], p["sd"])
    if t == "truncated_normal":
        if p["sd"] <= 0 or not p["low"] < p["high"]:
            raise MonteCarloError("truncated_normal needs sd > 0 and low < high")
        return stats.truncnorm((p["low"] - p["mean"]) / p["sd"], (p["high"] - p["mean"]) / p["sd"],
                               loc=p["mean"], scale=p["sd"])
    if t == "triangular":
        if not p["low"] <= p["mode"] <= p["high"] or p["low"] == p["high"]:
            raise MonteCarloError("triangular needs low <= mode <= high and low < high")
        w = p["high"] - p["low"]
        return stats.triang((p["mode"] - p["low"]) / w, loc=p["low"], scale=w)
    if t == "uniform":
        if not p["low"] < p["high"]:
            raise MonteCarloError("uniform needs low < high")
        return stats.uniform(p["low"], p["high"] - p["low"])
    return _Fixed(p["value"])


@dataclass
class MCSpec:
    distributions: dict            # name -> spec dict, for all four DRAWN names
    n_sims: int = 10_000
    seed: int = 12345
    corr_cagr_margin: float = 0.0  # Gaussian-copula correlation of the (CAGR, margin) pair

    def dists(self) -> dict:
        missing = [n for n in DRAWN if n not in self.distributions]
        if missing:
            raise MonteCarloError(f"missing distributions for {missing}")
        return {n: make_dist(self.distributions[n]) for n in DRAWN}


def spec_from_config(cc: dict) -> MCSpec:
    """Build an MCSpec from a company config block (`monte_carlo:` in config/companies.yaml)."""
    mc = cc.get("monte_carlo")
    if not mc:
        raise MonteCarloError("no monte_carlo block in config")
    rho = mc.get("correlation_cagr_margin", {}).get("value")
    if rho is None:
        raise MonteCarloError("monte_carlo.correlation_cagr_margin.value is unset")
    return MCSpec(distributions=mc["distributions"], n_sims=int(mc.get("n_sims", 10_000)),
                  seed=int(mc["seed"]), corr_cagr_margin=float(rho))


def probability_ge(values, price: float) -> tuple[float, float]:
    """P(value >= price) and its binomial standard error sqrt(p(1-p)/n)."""
    v = np.asarray(values, dtype=float)
    p = float(np.mean(v >= price))
    return p, float(np.sqrt(p * (1 - p) / len(v)))


@dataclass
class MCResult:
    price: float
    seed: int
    n_sims: int
    corr_cagr_margin: float
    values: np.ndarray                  # valid draws' value per share
    draws: pd.DataFrame                 # accepted draws (n_sims rows), aligned with `all_values`
    all_values: np.ndarray              # NaN where the DCF errored
    stats: dict
    prob_ge_price: float
    prob_se: float
    n_valid: int
    n_errors: int
    error_breakdown: dict
    n_draws_consumed: int               # raw draws needed to obtain n_sims accepted ones
    n_rejected_g_ge_wacc: int
    rejection_share: float
    accepted_vs_unconditional: dict = field(default_factory=dict)
    held_fixed: dict = field(default_factory=dict)
    distributions: dict = field(default_factory=dict)

    def report(self) -> str:
        s = self.stats
        L = [f"MONTE CARLO  n={self.n_sims:,} seed={self.seed} corr(CAGR,margin)={self.corr_cagr_margin:+.2f}",
             f"  price {self.price:,.2f}   P(value >= price) = {self.prob_ge_price:.4f} "
             f"(+/- {self.prob_se:.4f} s.e.)",
             "  value/share: mean {mean:,.2f}  median {p50:,.2f}  p5 {p5:,.2f}  p25 {p25:,.2f}  "
             "p75 {p75:,.2f}  p95 {p95:,.2f}".format(**s),
             f"  g >= WACC rejected & redrawn: {self.n_rejected_g_ge_wacc:,} of "
             f"{self.n_draws_consumed:,} raw draws ({self.rejection_share:.2%})"]
        for k, v in self.accepted_vs_unconditional.items():
            L.append(f"    {k}: unconditional mean {v['unconditional']:.4f} -> accepted mean {v['accepted']:.4f}")
        L.append(f"  DCF errors (excluded, counted): {self.n_errors} {self.error_breakdown or ''}")
        L.append("  distributions (ASSUMPTIONS): " + str(self.distributions))
        L.append("  held fixed: " + str(self.held_fixed))
        return "\n".join(L)

    __str__ = report


def _draw_accepted(spec: MCSpec, rho: float):
    """Draw until n_sims draws satisfy g < WACC. Returns (accepted df, unconditional df, consumed)."""
    dists = spec.dists()
    n = spec.n_sims
    if not -1.0 < rho < 1.0:
        raise MonteCarloError("correlation must be strictly between -1 and 1")
    rng = np.random.default_rng(spec.seed)
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    acc_parts, all_parts, have, consumed = [], [], 0, 0
    while have < n:
        batch = max(n - have, 1000) * 2
        z = rng.standard_normal((batch, 4))
        z[:, :2] = z[:, :2] @ chol.T                       # correlate CAGR & margin only
        u = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
        x = np.column_stack([dists[name].ppf(u[:, _COLS[name]]) for name in DRAWN])
        df = pd.DataFrame(x, columns=DRAWN)
        ok = (df["terminal_growth"] < df["wacc"]).to_numpy()
        # take rows in order; stop at the row that completes n accepted draws
        need = n - have
        cum = np.cumsum(ok)
        stop = int(np.searchsorted(cum, need)) + 1 if cum[-1] >= need else batch
        all_parts.append(df.iloc[:stop])
        acc_parts.append(df.iloc[:stop][ok[:stop]])
        have += int(ok[:stop].sum())
        consumed += stop
    accepted = pd.concat(acc_parts, ignore_index=True).iloc[:n]
    unconditional = pd.concat(all_parts, ignore_index=True)
    return accepted, unconditional, consumed


def run_monte_carlo(base: dcf.DCFInputs, spec: MCSpec, price: float,
                    corr: float | None = None) -> MCResult:
    """Run n_sims draws through the forward DCF. `corr` overrides spec.corr_cagr_margin."""
    rho = spec.corr_cagr_margin if corr is None else corr
    accepted, uncond, consumed = _draw_accepted(spec, rho)

    vals = np.full(len(accepted), np.nan)
    errors: Counter = Counter()
    for i, row in enumerate(accepted.itertuples(index=False)):
        try:
            vals[i] = dcf.value_per_share(base.with_(**dict(zip(DRAWN, row))))
        except Exception as e:                      # counted and reported, never silently dropped
            errors[type(e).__name__ + ": " + str(e)[:60]] += 1
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        raise MonteCarloError(f"every draw failed inside the DCF: {dict(errors)}")

    pcts = np.percentile(valid, [5, 25, 50, 75, 95])
    st = {"mean": float(valid.mean()), "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
          "p5": float(pcts[0]), "p25": float(pcts[1]), "p50": float(pcts[2]),
          "p75": float(pcts[3]), "p95": float(pcts[4])}
    p, se = probability_ge(valid, price)
    from src.reverse_solver import held_fixed
    return MCResult(
        price=price, seed=spec.seed, n_sims=spec.n_sims, corr_cagr_margin=rho, values=valid,
        draws=accepted, all_values=vals, stats=st, prob_ge_price=p, prob_se=se,
        n_valid=len(valid), n_errors=int(sum(errors.values())), error_breakdown=dict(errors),
        n_draws_consumed=consumed, n_rejected_g_ge_wacc=consumed - spec.n_sims,
        rejection_share=(consumed - spec.n_sims) / consumed,
        accepted_vs_unconditional={k: {"unconditional": float(uncond[k].mean()),
                                       "accepted": float(accepted[k].mean())}
                                   for k in ("wacc", "terminal_growth")},
        held_fixed=held_fixed(base, exclude=DRAWN),
        distributions={n: spec.distributions[n] for n in DRAWN})


def compare_correlation(base: dcf.DCFInputs, spec: MCSpec, price: float) -> dict[str, MCResult]:
    """Same seed and random numbers, correlation 0 vs the spec's correlation."""
    return {"uncorrelated": run_monte_carlo(base, spec, price, corr=0.0),
            "correlated": run_monte_carlo(base, spec, price, corr=spec.corr_cagr_margin)}
