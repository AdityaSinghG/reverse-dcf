"""Textbook DCF checks, hand-calculable, no company data."""
import pytest

from src import dcf


def test_three_year_hand_example():
    # FCFF 100, 110, 121; WACC 10%; g 3%.
    # PV flows = 3 x 90.9091 = 272.7273; TV = 121*1.03/0.07 = 1780.4286; PV(TV) = 1337.6624
    ev = dcf.enterprise_value([100, 110, 121], wacc=0.10, g=0.03)
    assert ev == pytest.approx(272.727273 + 1337.662364, abs=0.01)
    assert ev == pytest.approx(1610.39, abs=0.01)


def test_growing_perpetuity_closed_form():
    # Flows growing exactly at g over 10 years then Gordon TV == plain Gordon on flow 1: 100/(0.10-0.05)
    flows = [100 * 1.05 ** k for k in range(10)]
    assert dcf.enterprise_value(flows, wacc=0.10, g=0.05) == pytest.approx(2000.0, rel=1e-12)


def test_mid_year_is_exactly_half_year_uplift():
    flows = [50, 60, 70, 80]
    end = dcf.enterprise_value(flows, 0.09, 0.03, mid_year=False)
    mid = dcf.enterprise_value(flows, 0.09, 0.03, mid_year=True)
    assert mid == pytest.approx(end * 1.09 ** 0.5, rel=1e-12)


@pytest.mark.parametrize("g", [0.10, 0.12])
def test_g_must_be_below_wacc(g):
    with pytest.raises(dcf.ValuationError):
        dcf.gordon_terminal_value(100, wacc=0.10, g=g)
    with pytest.raises(dcf.ValuationError):
        dcf.enterprise_value([1, 2, 3], wacc=0.10, g=g)


def test_discount_factors_values():
    assert dcf.discount_factors(0.10, 2).tolist() == pytest.approx([1 / 1.1, 1 / 1.21])
    assert dcf.discount_factors(0.10, 1, mid_year=True)[0] == pytest.approx(1.1 ** -0.5)
