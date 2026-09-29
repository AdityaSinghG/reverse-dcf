"""Plot smoke tests on TEST (synthetic) inputs: files are produced, sources are mandatory."""
import numpy as np
import pytest

from src import dcf, monte_carlo as mc, plots, reverse_solver as rs, sensitivity as sens
from tests.test_monte_carlo import base, spec

SRC = "TEST: synthetic inputs, not company data"


def _png(p):
    return p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 5_000


def test_cagr_helper():
    assert plots.cagr(100, 200, 5) == pytest.approx(2 ** 0.2 - 1)
    with pytest.raises(ValueError):
        plots.cagr(-1, 5, 3)


def test_all_four_charts_render(tmp_path):
    b = base()
    price = dcf.value_per_share(b)
    p1 = plots.plot_implied_vs_historical_cagr(
        0.15, {"Hist 3y": 0.42, "Hist 5y": 0.30, "Peer A 5y": 0.12}, title="Implied vs historical CAGR",
        source=SRC, path=tmp_path / "a.png")
    fr = rs.implied_frontier(b, price, np.linspace(0.10, 0.28, 12), n_scan=120)
    p2 = plots.plot_frontier(fr, title="Frontier", source=SRC, margin_band=(0.12, 0.22, "Peer margin range"),
                             cagr_band=(0.10, 0.14, "Hist range"), points={"Base": (0.15, 0.18)},
                             path=tmp_path / "b.png")
    both = mc.compare_correlation(b, spec(n=3000, rho=0.5), price)
    p3 = plots.plot_mc_histogram(both["correlated"], uncorrelated=both["uncorrelated"],
                                 title="MC", source=SRC, path=tmp_path / "c.png")
    tor = sens.tornado(b, sens.ranges_from_spec(spec()))
    p4 = plots.plot_tornado(tor, title="Tornado", source=SRC, price=price, path=tmp_path / "d.png")
    assert all(_png(p) for p in (p1, p2, p3, p4))


def test_source_note_is_mandatory(tmp_path):
    with pytest.raises(ValueError, match="source"):
        plots.plot_implied_vs_historical_cagr(0.1, {"x": 0.2}, title="t", source="  ",
                                              path=tmp_path / "x.png")
