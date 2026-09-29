"""Forecast engine checks against closed forms and hand calculations."""
import numpy as np
import pytest

from src import dcf


def base(**kw):
    d = dict(base_revenue=1000.0, base_ebit_margin=0.20, revenue_cagr=0.05, terminal_growth=0.05,
             terminal_ebit_margin=0.20, wacc=0.11, sales_to_capital=2.0, tax_rate=0.25,
             tax_rate_terminal=0.25, opening_nol=0.0, years=10)
    d.update(kw)
    return dcf.DCFInputs(**d)


def test_engine_matches_steady_state_closed_form():
    # Constant margin, growth = terminal growth, no NOL -> FCFF_t = F1 (1+g)^(t-1), EV = F1/(WACC-g)
    inp = base()
    g, m, t, s2c, r0 = 0.05, 0.20, 0.25, 2.0, 1000.0
    f1 = r0 * ((1 + g) * m * (1 - t) - g / s2c)
    res = dcf.run_dcf(inp)
    assert res.enterprise_value == pytest.approx(f1 / (0.11 - g), rel=1e-12)


def test_engine_first_year_hand_numbers():
    res = dcf.run_dcf(base(revenue_cagr=0.10, terminal_ebit_margin=0.20))
    y1 = res.forecast.iloc[1]
    assert y1["revenue"] == pytest.approx(1100)
    assert y1["ebit"] == pytest.approx(220)
    assert y1["cash_tax"] == pytest.approx(55)          # 25% x 220
    assert y1["reinvestment"] == pytest.approx(50)      # (1100-1000)/2
    assert y1["fcff"] == pytest.approx(220 - 55 - 50)


def test_nol_roll_forward_hand():
    tax, o, used, added, close = dcf.tax_schedule(np.array([-20.0, 10.0, 30.0, 50.0]), 0.25, 5.0)
    assert o.tolist() == [5, 25, 15, 0]
    assert used.tolist() == [0, 10, 15, 0]
    assert added.tolist() == [20, 0, 0, 0]
    assert close.tolist() == [25, 15, 0, 0]
    assert tax.tolist() == pytest.approx([0, 0, 3.75, 12.5])


def test_nol_shelters_tax_and_adds_value():
    no_nol = dcf.run_dcf(base(base_ebit_margin=-0.10, opening_nol=0.0, terminal_ebit_margin=0.15))
    with_nol = dcf.run_dcf(base(base_ebit_margin=-0.10, opening_nol=500.0, terminal_ebit_margin=0.15))
    assert with_nol.enterprise_value > no_nol.enterprise_value
    # tax never negative and cumulative NOL usage never exceeds pool
    f = with_nol.forecast.iloc[1:]
    assert (f["cash_tax"] >= 0).all()
    assert f["nol_used"].sum() <= 500 + f["nol_added"].sum() + 1e-9


def test_unused_nol_valued_as_residual_shield():
    # Permanent losses in explicit period leave a large pool at N
    res = dcf.run_dcf(base(base_ebit_margin=-0.5, terminal_ebit_margin=-0.5 + 1e-9,
                           terminal_growth=0.02, revenue_cagr=0.02))
    nol_n = res.forecast["nol_close"].iloc[-1]
    assert nol_n > 0
    df_n = res.forecast["discount_factor"].iloc[-1]
    assert res.pv_residual_nol == pytest.approx(0.25 * nol_n * df_n)


def test_terminal_tax_rate_convergence_lowers_value():
    lo = dcf.run_dcf(base(tax_rate_terminal=0.25))
    hi = dcf.run_dcf(base(tax_rate_terminal=0.30))
    assert hi.enterprise_value < lo.enterprise_value


def test_margin_ramp_reaches_terminal_and_holds():
    m = dcf.margin_path(-0.05, 0.15, years=10, ramp_years=5)
    assert m[0] == pytest.approx(-0.05 + 0.04)
    assert m[4:] == pytest.approx(0.15)


def test_growth_profiles():
    assert dcf.growth_path(0.2, 0.05, 10, "constant").tolist() == [0.2] * 10
    lf = dcf.growth_path(0.25, 0.05, 10, "linear_fade")
    assert lf[0] == pytest.approx(0.25) and lf[-1] == pytest.approx(0.05)
    assert np.all(np.diff(lf) < 0)
    with pytest.raises(dcf.ValuationError):
        dcf.growth_path(0.2, 0.05, 10, "nope")


def test_realised_cagr_constant():
    assert dcf.run_dcf(base(revenue_cagr=0.13)).realised_cagr == pytest.approx(0.13)


def test_bridge_and_per_share():
    vals = {"cash": 100.0, "inv": 50.0, "nci": 5.0, "debt": 30.0, "lease": 20.0, "noncurrent_inv": 40.0}
    b = dcf.build_bridge(vals, {"cash": 1, "inv": 1, "noncurrent_inv": 0}, ["debt", "lease", "nci"])
    assert b["noncurrent_inv"] == 0 and b["lease"] == -20
    assert sum(b.values()) == pytest.approx(100 + 50 - 30 - 20 - 5)
    res = dcf.run_dcf(base(bridge=b, diluted_shares=10.0))
    assert res.equity_value == pytest.approx(res.enterprise_value + 95)
    assert res.value_per_share == pytest.approx(res.equity_value / 10)


def test_bridge_missing_item_raises_not_zero():
    with pytest.raises(dcf.ValuationError):
        dcf.build_bridge({"cash": 1.0}, {"cash": 1, "inv": 1}, [])


def test_operating_ebit_excludes_finance_costs_and_other_income():
    # revenue 1000; expenses 1100 incl. finance cost 20 (lease interest) -> EBIT -80.
    # Other income is not in total_expenses and is deliberately not added.
    assert dcf.operating_ebit(1000, 1100, 20) == -80


def test_engine_guards():
    with pytest.raises(dcf.ValuationError):
        dcf.run_dcf(base(terminal_growth=0.11))
    with pytest.raises(dcf.ValuationError):
        dcf.run_dcf(base(sales_to_capital=0))
    with pytest.raises(dcf.ValuationError):
        dcf.run_dcf(base(diluted_shares=0))
    with pytest.raises(dcf.ValuationError):
        dcf.run_dcf(base(margin_ramp_years=11))


def test_mid_year_uplift_through_engine():
    e = dcf.run_dcf(base(mid_year=False, opening_nol=0))
    m = dcf.run_dcf(base(mid_year=True, opening_nol=0))
    assert m.enterprise_value == pytest.approx(e.enterprise_value * 1.11 ** 0.5, rel=1e-12)


def test_marginal_roic_reported():
    res = dcf.run_dcf(base())
    assert res.terminal_marginal_roic == pytest.approx(2.0 * 0.20 * 0.75)


@pytest.mark.parametrize("kw", [
    {},                                                                    # steady state
    dict(base_ebit_margin=-0.10, terminal_ebit_margin=0.15, revenue_cagr=0.18),          # loss start
    dict(base_ebit_margin=-0.30, terminal_ebit_margin=0.08, opening_nol=800.0, wacc=0.11,
         sales_to_capital=2.0),                                            # loss pool in use
    dict(base_ebit_margin=-0.50, terminal_ebit_margin=-0.5 + 1e-9, terminal_growth=0.02,
         revenue_cagr=0.02),                                               # pool never exhausted
    dict(opening_nol=5000.0, base_ebit_margin=0.05, terminal_ebit_margin=0.20),          # big pool, profitable
    dict(growth_profile="linear_fade", revenue_cagr=0.30, mid_year=True, margin_ramp_years=6,
         base_ebit_margin=-0.15, opening_nol=300.0),
    dict(revenue_cagr=-0.03, terminal_growth=0.01, tax_rate_terminal=0.30),              # shrinking revenue
    dict(bridge={"cash": 250.0, "debt": -40.0, "lease": -15.0}, diluted_shares=7.3),
])
def test_fast_path_equals_full_path_exactly(kw):
    inp = base(**kw)
    full = dcf.run_dcf(inp)
    fast = dcf.run_dcf(inp, build_table=False)
    assert fast.forecast is None and full.forecast is not None
    assert fast.value_per_share == full.value_per_share            # bit-for-bit, not approx
    assert fast.enterprise_value == full.enterprise_value
    assert fast.pv_residual_nol == full.pv_residual_nol
    assert dcf.value_per_share(inp) == full.value_per_share
