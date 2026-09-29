import numpy as np
import pandas as pd
import pytest

from src import wacc as w


def test_capm_hand():
    assert w.cost_of_equity(0.07, 1.2, 0.07) == pytest.approx(0.154)


def test_wacc_hand():
    # Ke 15%, Kd pre-tax 9% at t=25% -> 6.75%; weights 80/20 -> 0.8*15 + 0.2*6.75 = 13.35%
    kd = w.after_tax_cost_of_debt(0.09, 0.25)
    assert kd == pytest.approx(0.0675)
    e, d = w.capital_weights(800, 200)
    assert (e, d) == pytest.approx((0.8, 0.2))
    assert w.wacc(0.15, kd, e, d) == pytest.approx(0.1335)


def test_weights_must_sum_to_one():
    with pytest.raises(w.WaccError):
        w.wacc(0.1, 0.05, 0.6, 0.6)


def test_zero_debt_wacc_equals_ke():
    e, d = w.capital_weights(1000, 0)
    assert w.wacc(0.14, 0.05, e, d) == pytest.approx(0.14)


def test_unlever_relever_roundtrip_and_hand():
    bu = w.unlever_beta(1.5, debt=50, equity=100, tax_rate=0.25)  # 1.5 / (1 + .75*.5)
    assert bu == pytest.approx(1.5 / 1.375)
    assert w.relever_beta(bu, 50, 100, 0.25) == pytest.approx(1.5)


def test_blume():
    assert w.blume_adjust(1.5) == pytest.approx(4 / 3)


def _prices_from_returns(r, start="2021-01-01", freq="W-FRI"):
    idx = pd.date_range(start, periods=len(r) + 1, freq=freq)
    return pd.Series(100 * np.cumprod(np.r_[1.0, 1 + r]), index=idx)


def test_regression_recovers_known_beta():
    rng = np.random.default_rng(0)
    rm = rng.normal(0.002, 0.02, 300)
    rs = 0.001 + 1.6 * rm  # exact linear: beta 1.6, R^2 = 1
    res = w.regression_beta(_prices_from_returns(rs), _prices_from_returns(rm),
                            freq="W", window_years=None)
    assert res.beta == pytest.approx(1.6, abs=1e-9)
    assert res.r_squared == pytest.approx(1.0, abs=1e-9)
    assert res.n_obs == 300


def test_regression_with_noise_and_window():
    rng = np.random.default_rng(1)
    rm = rng.normal(0.0, 0.02, 400)
    rs = 0.9 * rm + rng.normal(0, 0.01, 400)
    res = w.regression_beta(_prices_from_returns(rs), _prices_from_returns(rm),
                            freq="W", window_years=2)
    assert 103 <= res.n_obs <= 105              # ~2y of weeks
    lo, hi = res.ci95
    assert lo < 0.9 < hi and res.std_err > 0


def test_too_few_obs_raises():
    rm = np.random.default_rng(2).normal(0, 0.02, 10)
    with pytest.raises(w.WaccError):
        w.regression_beta(_prices_from_returns(rm), _prices_from_returns(rm), window_years=None)


def test_price_file_requires_provenance(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("date,close\n2024-01-05,100\n", encoding="utf-8")
    with pytest.raises(w.WaccError, match="provenance"):
        w.load_price_series(p)
    p.write_text("# source: TEST FIXTURE\n# retrieved: 2025-01-01\ndate,close\n2024-01-05,100\n",
                 encoding="utf-8")
    assert w.load_price_series(p).iloc[0] == 100


def test_peer_check_median_and_relever():
    peers = pd.DataFrame({
        "name": ["A", "B", "C"], "levered_beta": [1.2, 1.0, 1.4],
        "debt": [0, 0, 0], "market_cap": [100, 100, 100], "tax_rate": [0.25] * 3,
        "source": ["TEST FIXTURE"] * 3, "source_date": ["2025-01-01"] * 3})
    out = w.peer_beta_check(peers, target_debt=20, target_equity=100, target_tax=0.25)
    last = out.iloc[-1]
    assert last["unlevered_beta"] == pytest.approx(1.2)
    assert last["levered_beta"] == pytest.approx(1.2 * (1 + 0.75 * 0.2))
