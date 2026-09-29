"""Round-trip and edge-case tests for the reverse solver and sensitivity grids.
All inputs are synthetic; nothing here is Eternal data."""
import numpy as np
import pytest

from src import dcf, reverse_solver as rs, sensitivity as sens


def known(**kw):
    """Loss-making start, big NOL, net cash: same shape as the real problem."""
    d = dict(base_revenue=1000.0, base_ebit_margin=-0.10, revenue_cagr=0.18,
             terminal_growth=0.05, terminal_ebit_margin=0.20, wacc=0.12, sales_to_capital=2.5,
             tax_rate=0.25, tax_rate_terminal=0.25, opening_nol=500.0, years=10,
             bridge={"cash": 300.0, "debt": -80.0}, diluted_shares=10.0)
    d.update(kw)
    return dcf.DCFInputs(**d)


@pytest.mark.parametrize("param,true_value", [
    ("revenue_cagr", 0.18), ("terminal_ebit_margin", 0.20), ("wacc", 0.12)])
def test_round_trip_recovers_input(param, true_value):
    inp = known()
    price = dcf.value_per_share(inp)                    # forward: inputs -> price
    start = inp.with_(**{param: true_value * 0.5 + 0.01})   # solver must ignore the starting value
    res = rs.solve_implied(start, price, param)          # reverse: price -> inputs
    assert res.status == rs.SOLVED
    assert res.value == pytest.approx(true_value, abs=1e-9)
    # and the solved value reproduces the price through the forward model
    assert dcf.value_per_share(start.with_(**{param: res.value})) == pytest.approx(price, rel=1e-9)


def test_round_trip_with_linear_fade_and_mid_year():
    inp = known(growth_profile="linear_fade", mid_year=True, margin_ramp_years=6)
    price = dcf.value_per_share(inp)
    res = rs.solve_implied(inp.with_(revenue_cagr=0.3), price, "revenue_cagr")
    assert res.value == pytest.approx(0.18, abs=1e-9)


def test_no_solution_in_range_is_reported_not_widened():
    inp = known()
    res = rs.solve_implied(inp, price=1e9, parameter="revenue_cagr")
    assert res.status == rs.NO_SOLUTION and res.value is None and res.roots == []
    assert res.bracket_used == res.bracket_requested == rs.DEFAULT_BRACKETS["revenue_cagr"]
    assert "no solution in range" in res.report()


def test_custom_bracket_is_respected():
    inp = known()
    price = dcf.value_per_share(inp)
    res = rs.solve_implied(inp, price, "revenue_cagr", bracket=(0.20, 0.40))  # true 0.18 is outside
    assert res.status == rs.NO_SOLUTION and res.bracket_used == (0.20, 0.40)


def test_wacc_floor_guard_is_disclosed():
    inp = known()
    price = dcf.value_per_share(inp)
    res = rs.solve_implied(inp, price, "wacc", bracket=(0.01, 0.30))
    assert res.status == rs.SOLVED and res.value == pytest.approx(0.12, abs=1e-9)
    assert res.bracket_used[0] == pytest.approx(0.05 + 1e-4)      # raised to g + 1bp*... disclosed
    assert res.bracket_requested == (0.01, 0.30)
    assert any("raised" in n for n in res.notes)


def test_multiple_roots_reported_from_loss_pool_case():
    # Deep-loss start, modest terminal margin: value FALLS then RISES as CAGR increases
    # (bigger growth deepens early losses before late-year profits repay them), so a price
    # between the trough and the low-growth value is hit twice.
    inp = known(base_ebit_margin=-0.30, terminal_ebit_margin=0.08, sales_to_capital=2.0, wacc=0.11,
                opening_nol=0.0, bridge={"cash": 3000.0}, diluted_shares=1.0)
    xs = np.linspace(-0.05, 0.60, 400)
    vals = np.array([dcf.value_per_share(inp.with_(revenue_cagr=x)) for x in xs])
    trough, start = vals.min(), vals[0]
    assert trough > 0
    assert trough < start - 1                        # shape assumption behind the test
    price = trough + 0.5 * (start - trough)
    res = rs.solve_implied(inp, price, "revenue_cagr")
    assert res.status == rs.MULTIPLE and res.value is None and len(res.roots) == 2
    for r in res.roots:
        assert dcf.value_per_share(inp.with_(revenue_cagr=r)) == pytest.approx(price, rel=1e-9)
    assert any("non-monotone" in n for n in res.notes)


def test_report_prints_held_fixed_assumptions():
    inp = known()
    res = rs.solve_implied(inp, dcf.value_per_share(inp), "terminal_ebit_margin")
    text = res.report()
    for expected in ("held fixed", "wacc", "terminal_growth", "sales_to_capital", "opening_nol",
                     "tax_rate_terminal", "diluted_shares", "bridge_net_claims", "revenue_cagr"):
        assert expected in text
    assert "terminal_ebit_margin  " not in text.split("held fixed:")[1]   # solved param not listed as fixed


def test_solve_all_returns_three_results():
    inp = known()
    price = dcf.value_per_share(inp)
    out = rs.solve_all(inp, price)
    assert set(out) == set(rs.PARAMETERS)
    assert all(r.status == rs.SOLVED for r in out.values())


def test_bad_inputs_rejected():
    inp = known()
    with pytest.raises(ValueError):
        rs.solve_implied(inp, 100.0, "sales_to_capital")
    with pytest.raises(ValueError):
        rs.solve_implied(inp, -1.0, "wacc")
    with pytest.raises(ValueError):
        rs.solve_implied(inp, 100.0, "wacc", bracket=(0.10, 0.05))


def test_frontier_points_all_reproduce_price():
    inp = known()
    price = dcf.value_per_share(inp)
    cagrs = np.linspace(0.10, 0.28, 10)
    fr = rs.implied_frontier(inp, price, cagrs, n_scan=150)
    ok = fr[fr["status"] == rs.SOLVED]
    assert len(ok) >= 8
    for r in ok.itertuples():
        v = dcf.value_per_share(inp.with_(revenue_cagr=r.revenue_cagr,
                                          terminal_ebit_margin=r.implied_terminal_margin))
        assert v == pytest.approx(price, rel=1e-8)
    # trade-off: more growth needed less margin (downward-sloping frontier here)
    assert ok["implied_terminal_margin"].is_monotonic_decreasing
    # the known point lies on the frontier
    at = rs.solve_implied(inp.with_(revenue_cagr=0.18), price, "terminal_ebit_margin")
    assert at.value == pytest.approx(0.20, abs=1e-9)
    assert "held_fixed" in fr.attrs and "wacc" in fr.attrs["held_fixed"]


def test_frontier_marks_gaps():
    inp = known()
    fr = rs.implied_frontier(inp, price=1e6, cagrs=[0.1, 0.2])
    assert (fr["status"] == rs.NO_SOLUTION).all() and fr["implied_terminal_margin"].isna().all()


# ---- sensitivity ------------------------------------------------------------------------
def test_grid_center_cell_equals_forward_value():
    inp = known()
    g = sens.wacc_vs_terminal_g(inp, [0.10, 0.12, 0.14], [0.03, 0.05, 0.07])
    assert g.loc[0.12, 0.05] == pytest.approx(dcf.value_per_share(inp))
    assert g.loc[0.10, 0.03] > g.loc[0.14, 0.03]          # lower WACC -> higher value
    assert g.loc[0.12, 0.07] > g.loc[0.12, 0.03]          # higher g -> higher value (ROIC > WACC here)


def test_grid_invalid_cells_are_nan_not_zero():
    inp = known()
    g = sens.wacc_vs_terminal_g(inp, [0.06, 0.12], [0.05, 0.07])
    assert np.isnan(g.loc[0.06, 0.07])                    # g > WACC
    assert not np.isnan(g.loc[0.12, 0.07])
    assert np.isnan(sens.wacc_vs_terminal_g(inp, [0.05], [0.05]).iloc[0, 0])   # g == WACC


def test_cagr_vs_margin_grid_shape_monotonicity_and_premium():
    inp = known()
    price = dcf.value_per_share(inp)
    g = sens.cagr_vs_terminal_margin(inp, [0.14, 0.18, 0.22], [0.15, 0.20, 0.25], price=price)
    assert g.shape == (3, 3) and g.attrs["metric"] == "value/price - 1"
    assert g.loc[0.18, 0.20] == pytest.approx(0.0, abs=1e-12)     # at the known point value == price
    assert (np.diff(g.to_numpy(), axis=1) > 0).all()              # more margin -> more value
    assert "wacc" in g.attrs["held_fixed"]
