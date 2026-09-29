"""Monte Carlo tests. All inputs are TEST (synthetic), not Eternal data."""
import numpy as np
import pytest
from scipy import stats

from src import dcf, monte_carlo as mc, reverse_solver as rs, sensitivity as sens


def base(**kw):
    d = dict(base_revenue=1000.0, base_ebit_margin=-0.10, revenue_cagr=0.15,
             terminal_growth=0.05, terminal_ebit_margin=0.18, wacc=0.12, sales_to_capital=2.5,
             tax_rate=0.25, tax_rate_terminal=0.25, opening_nol=500.0, years=10,
             bridge={"cash": 300.0, "debt": -80.0}, diluted_shares=10.0)
    d.update(kw)
    return dcf.DCFInputs(**d)


def spec(n=2000, seed=7, rho=0.0, **over):
    dists = {
        "revenue_cagr": {"type": "triangular", "low": 0.08, "mode": 0.15, "high": 0.25},
        "terminal_ebit_margin": {"type": "triangular", "low": 0.10, "mode": 0.18, "high": 0.26},
        "wacc": {"type": "truncated_normal", "mean": 0.12, "sd": 0.01, "low": 0.09, "high": 0.15},
        "terminal_growth": {"type": "triangular", "low": 0.03, "mode": 0.05, "high": 0.06},
    }
    dists.update(over)
    return mc.MCSpec(distributions=dists, n_sims=n, seed=seed, corr_cagr_margin=rho)


# (a) reproducibility ------------------------------------------------------------------------
def test_seeded_run_reproduces_identically():
    a = mc.run_monte_carlo(base(), spec(), price=100.0)
    b = mc.run_monte_carlo(base(), spec(), price=100.0)
    assert np.array_equal(a.all_values, b.all_values, equal_nan=True)
    assert a.stats == b.stats and a.prob_ge_price == b.prob_ge_price
    assert a.seed == 7
    c = mc.run_monte_carlo(base(), spec(seed=8), price=100.0)
    assert not np.array_equal(a.all_values, c.all_values, equal_nan=True)


# (b) degenerate ------------------------------------------------------------------------------
def test_zero_variance_returns_deterministic_dcf_exactly():
    b = base()
    fixed = {n: {"type": "fixed", "value": getattr(b, n)} for n in mc.DRAWN}
    res = mc.run_monte_carlo(b, mc.MCSpec(distributions=fixed, n_sims=500, seed=1), price=50.0)
    det = dcf.value_per_share(b)
    assert (res.values == det).all()
    assert res.stats["std"] == pytest.approx(0.0, abs=1e-9)   # float noise in the mean only
    assert res.stats["p5"] == res.stats["p95"] == det
    assert res.n_errors == 0 and res.n_rejected_g_ge_wacc == 0


# (c) g < WACC ---------------------------------------------------------------------------------
def test_g_below_wacc_on_every_accepted_draw_and_rejections_reported():
    # overlapping wacc / g so a meaningful share of raw draws violates g < WACC
    s = spec(n=3000, wacc={"type": "normal", "mean": 0.07, "sd": 0.02},
             terminal_growth={"type": "normal", "mean": 0.05, "sd": 0.01})
    res = mc.run_monte_carlo(base(), s, price=100.0)
    assert (res.draws["terminal_growth"] < res.draws["wacc"]).all()
    assert len(res.draws) == 3000
    assert res.n_rejected_g_ge_wacc > 0
    assert res.n_draws_consumed == 3000 + res.n_rejected_g_ge_wacc
    assert res.rejection_share == pytest.approx(res.n_rejected_g_ge_wacc / res.n_draws_consumed)
    # bias is visible: accepted WACC mean above unconditional, accepted g mean below
    av = res.accepted_vs_unconditional
    assert av["wacc"]["accepted"] > av["wacc"]["unconditional"]
    assert av["terminal_growth"]["accepted"] < av["terminal_growth"]["unconditional"]
    assert "rejected & redrawn" in res.report()


def test_no_rejections_when_supports_do_not_overlap():
    res = mc.run_monte_carlo(base(), spec(), price=100.0)
    assert res.n_rejected_g_ge_wacc == 0


# (d) probability ---------------------------------------------------------------------------------
def test_probability_helper_known_values():
    assert mc.probability_ge([1, 2, 3, 4], 2.5) == (0.5, pytest.approx(0.25))
    assert mc.probability_ge([1, 2, 3, 4], 1.0)[0] == 1.0          # >= is inclusive
    assert mc.probability_ge([1, 2, 3, 4], 5.0)[0] == 0.0


def test_probability_matches_analytic_uniform_cagr():
    # Only CAGR varies, uniform(low, high); value is monotone increasing in CAGR, so
    # P(value >= price) = P(CAGR >= implied CAGR) = (high - c*) / (high - low), exactly known.
    b = base()
    fixed = {n: {"type": "fixed", "value": getattr(b, n)}
             for n in ("terminal_ebit_margin", "wacc", "terminal_growth")}
    lo, hi = 0.08, 0.28
    s = mc.MCSpec(distributions={**fixed, "revenue_cagr": {"type": "uniform", "low": lo, "high": hi}},
                  n_sims=20000, seed=3)
    price = dcf.value_per_share(b.with_(revenue_cagr=0.20))
    implied = rs.solve_implied(b, price, "revenue_cagr").value
    expected = (hi - implied) / (hi - lo)
    res = mc.run_monte_carlo(b, s, price)
    assert implied == pytest.approx(0.20, abs=1e-9)
    assert abs(res.prob_ge_price - expected) < 4 * res.prob_se


def test_dist_ppf_matches_scipy_and_bounds():
    d = mc.make_dist({"type": "truncated_normal", "mean": 0.1, "sd": 0.05, "low": 0.0, "high": 0.2})
    x = d.ppf(np.linspace(0.001, 0.999, 999))
    assert x.min() >= 0.0 and x.max() <= 0.2
    t = mc.make_dist({"type": "triangular", "low": 0.0, "mode": 0.1, "high": 0.4})
    assert t.ppf(0.5) == pytest.approx(stats.triang(0.25, 0, 0.4).ppf(0.5))


# (e) correlation ----------------------------------------------------------------------------------
def test_correlation_changes_spread_in_expected_direction():
    # Value rises with both CAGR and margin, so positive correlation stacks good/bad cases and
    # widens the distribution; negative correlation offsets them and narrows it.
    s = spec(n=6000)
    pos = mc.run_monte_carlo(base(), s, 100.0, corr=0.7)
    zero = mc.run_monte_carlo(base(), s, 100.0, corr=0.0)
    neg = mc.run_monte_carlo(base(), s, 100.0, corr=-0.7)
    assert pos.stats["std"] > zero.stats["std"] > neg.stats["std"]
    # realised rank correlation of the drawn inputs tracks the requested one
    r = stats.spearmanr(pos.draws["revenue_cagr"], pos.draws["terminal_ebit_margin"]).statistic
    assert r == pytest.approx(0.7, abs=0.06)
    assert abs(stats.spearmanr(zero.draws["revenue_cagr"], zero.draws["terminal_ebit_margin"]).statistic) < 0.06


def test_correlation_leaves_marginals_unchanged():
    s = spec(n=6000)
    pos = mc.run_monte_carlo(base(), s, 100.0, corr=0.7)
    zero = mc.run_monte_carlo(base(), s, 100.0, corr=0.0)
    for col in ("revenue_cagr", "terminal_ebit_margin"):
        assert pos.draws[col].mean() == pytest.approx(zero.draws[col].mean(), abs=0.003)
    # uncorrelated columns are literally the same draws (common random numbers)
    assert np.array_equal(pos.draws["wacc"], zero.draws["wacc"])


def test_compare_correlation_reports_both_probabilities():
    out = mc.compare_correlation(base(), spec(rho=0.5), price=100.0)
    assert set(out) == {"uncorrelated", "correlated"}
    assert out["uncorrelated"].corr_cagr_margin == 0.0 and out["correlated"].corr_cagr_margin == 0.5
    assert 0 <= out["correlated"].prob_ge_price <= 1


# errors are counted, never dropped ------------------------------------------------------------------
def test_dcf_errors_are_counted_not_dropped(monkeypatch):
    real = dcf.value_per_share
    calls = {"n": 0}

    def flaky(inp):
        calls["n"] += 1
        if calls["n"] % 10 == 0:
            raise dcf.ValuationError("synthetic failure")
        return real(inp)

    monkeypatch.setattr(mc.dcf, "value_per_share", flaky)
    res = mc.run_monte_carlo(base(), spec(n=500), price=100.0)
    assert res.n_errors == 50 and res.n_valid == 450
    assert res.n_valid + res.n_errors == res.n_sims
    assert sum(res.error_breakdown.values()) == 50 and "ValuationError" in next(iter(res.error_breakdown))
    assert np.isnan(res.all_values).sum() == 50
    assert "DCF errors" in res.report()


def test_all_draws_failing_raises(monkeypatch):
    monkeypatch.setattr(mc.dcf, "value_per_share",
                        lambda inp: (_ for _ in ()).throw(dcf.ValuationError("boom")))
    with pytest.raises(mc.MonteCarloError, match="every draw failed"):
        mc.run_monte_carlo(base(), spec(n=50), price=1.0)


# config / validation ---------------------------------------------------------------------------------
def test_unset_parameters_refuse_to_run():
    with pytest.raises(mc.MonteCarloError, match="unset"):
        mc.make_dist({"type": "triangular", "low": None, "mode": None, "high": None})
    with pytest.raises(mc.MonteCarloError):
        mc.make_dist({"type": "weibull"})


def test_repo_config_distributions_are_unset_until_user_decides():
    from src import data_loader as dl
    cc = dl.company_config("eternal")
    s = mc.spec_from_config(cc)                     # structure parses; seed and rho recorded
    assert s.seed == 20250101 and s.n_sims == 10000 and s.corr_cagr_margin == 0.3
    with pytest.raises(mc.MonteCarloError, match="unset"):
        s.dists()                                   # parameters are yours to set


def test_bad_correlation_rejected():
    with pytest.raises(mc.MonteCarloError):
        mc.run_monte_carlo(base(), spec(), 100.0, corr=1.0)


def test_tornado_orders_by_swing_and_uses_spec_ranges():
    s = spec()
    rg = sens.ranges_from_spec(s)
    assert rg["revenue_cagr"][0] == pytest.approx(s.dists()["revenue_cagr"].ppf(0.1))
    t = sens.tornado(base(), rg)
    assert t["swing"].is_monotonic_decreasing and set(t["driver"]) == set(mc.DRAWN)
    assert t.attrs["base_value"] == dcf.value_per_share(base())
