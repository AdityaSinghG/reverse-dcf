"""Rule 4: the forward DCF at solved parameters must reproduce the market price within 0.5%.

Two tests:
  * the TEST-data version always runs, so the test logic itself is proven;
  * the Eternal version is SKIPPED, with the reason listing what is missing, until the Eternal CSVs and
    assumptions (A1-A5, A7, A23-A27) exist. It then runs automatically on the real inputs.
"""
import pytest

from src import build_excel as be, company_bundle as cb, dcf, reverse_solver as rs

TOL = 0.005


def assert_reproduces_price(inp: dcf.DCFInputs, price: float, tol: float = TOL):
    """Solve each parameter, feed it back through the forward DCF, require value == price within tol.
    A parameter with no unique root cannot be tested and is reported, but at least one must solve."""
    results = rs.solve_all(inp, price)
    solved = {p: r for p, r in results.items() if r.status == rs.SOLVED}
    assert solved, "no parameter had a unique solution: " + ", ".join(f"{p}={r.status}" for p, r in results.items())
    for p, r in solved.items():
        v = dcf.value_per_share(inp.with_(**{p: r.value}))
        assert abs(v / price - 1) <= tol, f"forward DCF at implied {p}={r.value:.6f} gives {v:.4f}, price {price:.4f}"
    return results


def test_reproduction_logic_on_test_data():
    from tests.fixture_bundle import make_bundle
    b = make_bundle(n_sims=200)
    py = be.python_model(b)
    results = assert_reproduces_price(py["inp"], py["price"])
    assert all(r.status == rs.SOLVED for r in results.values())


def test_reproduction_test_would_fail_on_a_wrong_solution():
    from tests.fixture_bundle import make_bundle
    py = be.python_model(make_bundle(n_sims=200))
    inp = py["inp"]
    v_off = dcf.value_per_share(inp.with_(revenue_cagr=py["solved"]["revenue_cagr"].value + 0.01))
    assert abs(v_off / py["price"] - 1) > TOL           # a 1pt CAGR error is far outside tolerance


_rd = cb.check_readiness("eternal")


@pytest.mark.skipif(not _rd.ready, reason=(
    f"SKIPPED: Eternal inputs incomplete ({_rd.n} item(s) missing: " +
    "; ".join(f"{k} x{len(v)}" for k, v in _rd.problems.items()) +
    "). Run `python -m src.company_bundle eternal` for the exact list. This test runs automatically once complete."))
def test_eternal_forward_dcf_at_solved_parameters_reproduces_market_price():
    b = cb.load_bundle("eternal")
    py = be.python_model(b)
    assert_reproduces_price(py["inp"], py["price"])
