"""PRELIMINARY mode: labelled, price never invented, final-mode gates intact. LTM and tie-out logic."""
import copy

import pandas as pd
import pytest

from src import company_bundle as cb, data_loader as dl, tieouts as t, run_all
from tests import fixture_bundle as fx

LABEL = "PRELIMINARY - ILLUSTRATIVE ASSUMPTIONS"


def test_final_mode_gates_unchanged_by_the_preliminary_block():
    cc = cb.effective_config("eternal", preliminary=False)
    assert cc["assumptions"]["terminal_growth"]["value"] is None and "_preliminary" not in cc
    rd = cb.check_readiness("eternal", preliminary=False)
    assert not rd.ready and "A1:" in rd.text() and "A5:" in rd.text()


def test_preliminary_overlay_fills_assumptions_but_not_the_price():
    cc = cb.effective_config("eternal", preliminary=True)
    assert cc["assumptions"]["terminal_growth"]["value"] == 0.05 and cc["_label"] == LABEL
    assert cc["forecast"]["base"] == "LTM" and cc["first_fy"] == 2026
    rd = cb.check_readiness("eternal", preliminary=True)
    assert set(rd.problems) == {"market price (PRELIMINARY)"}          # nothing else blocks, and the price is not invented
    assert "Not invented" in rd.text()


def test_price_supplied_makes_it_ready_and_bundle_is_labelled():
    p = {"value": 123.0, "date": "2026-09-29"}
    assert cb.check_readiness("eternal", preliminary=True, price=p).ready
    b = cb.load_bundle("eternal", preliminary=True, price=p)
    assert b.label == LABEL and b.market["price"] == 123.0 and "APPROXIMATE" in b.market["source"]
    assert "esop_unamortised_cost" in b.placeholders and b.wacc_inputs["beta"]["source"].startswith("ILLUSTRATIVE")


def test_turning_preliminary_off_restores_refusal():
    p = {"value": 123.0, "date": "2026-09-29"}
    with pytest.raises(cb.IncompleteInputs):
        cb.load_bundle("eternal", preliminary=False, price=p)


def test_run_all_value_only_labels_every_line_and_skips_price_outputs(capsys):
    code = run_all.run("eternal", preliminary=True)
    out = capsys.readouterr().out
    assert code == 0
    prefixed = [l for l in out.splitlines() if l.startswith("[")]
    assert prefixed and all(l.startswith(f"[{LABEL}]") for l in prefixed)
    assert LABEL in out.splitlines()[0]
    for absent in ("[4] IMPLIED", "[5] MONTE CARLO", "[7] EXCEL", "vs price"):
        assert absent not in out
    assert "WAITING" in out


def test_run_all_final_mode_still_refuses(capsys):
    assert run_all.run("eternal", preliminary=False) == 2
    assert "CANNOT BUILD eternal" in capsys.readouterr().out


# ---- LTM ---------------------------------------------------------------------------------------------
def _df(rows):
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp()) / "x.csv"
    d.write_text(",".join(dl.LONG_COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return dl.read_long_table(d, "testco")


def _r(item, val, end, ptype):
    return f"testco,{ptype},{end},X,{item},{val},INR crore,T,2026-01-01,,"


def test_ltm_is_fy_plus_latest_quarter_minus_prior_year_quarter():
    df = _df([_r("rev", 54364, "2026-03-31", "FY"), _r("rev", 20211, "2026-06-30", "Q"), _r("rev", 7167, "2025-06-30", "Q")])
    assert dl.ltm_value(df, "rev") == 54364 + 20211 - 7167 == 67408


def test_ltm_refuses_without_the_prior_year_quarter_or_when_quarter_is_stale():
    with pytest.raises(dl.DataError, match="same quarter last year"):
        dl.ltm_value(_df([_r("rev", 1, "2026-03-31", "FY"), _r("rev", 2, "2026-06-30", "Q")]), "rev")
    with pytest.raises(dl.DataError, match="not within a year"):
        dl.ltm_value(_df([_r("rev", 1, "2026-03-31", "FY"), _r("rev", 2, "2025-12-31", "Q"), _r("rev", 3, "2024-12-31", "Q")]), "rev")


def test_ltm_prefers_four_consecutive_quarters():
    df = _df([_r("rev", v, e, "Q") for e, v in [("2025-09-30", 1), ("2025-12-31", 2), ("2026-03-31", 3), ("2026-06-30", 4)]]
             + [_r("rev", 99, "2026-03-31", "FY")])
    assert dl.ltm_value(df, "rev") == 10


# ---- tie-outs report failures and never adjust -----------------------------------------------------------------
def test_balance_sheet_tie_flags_a_planted_difference():
    df = _df([_r("total_assets", 100, "2026-03-31", "FY"), _r("total_equity", 60, "2026-03-31", "FY"),
              _r("total_liabilities", 41, "2026-03-31", "FY")])
    ties = t.balance_sheet_ties(df)
    assert len(ties) == 1 and ties[0].diff == -1.0 and ties[0].ok             # within the 1-unit rounding tolerance
    df2 = _df([_r("total_assets", 100, "2026-03-31", "FY"), _r("total_equity", 60, "2026-03-31", "FY"),
               _r("total_liabilities", 45, "2026-03-31", "FY")])
    bad = t.balance_sheet_ties(df2)[0]
    assert not bad.ok and bad.diff == -5.0 and "FAIL" in bad.line()
    assert "nothing was adjusted" in t.report([bad])


def test_segment_tie_flags_mismatch():
    cc = {"segments": ["a", "b"], "segment_tolerance_pct": 0.5}
    df = _df([_r("revenue_from_operations", 1000, "2026-03-31", "FY"), _r("segment_revenue__a", 600, "2026-03-31", "FY"),
              _r("segment_revenue__b", 300, "2026-03-31", "FY")])
    ties = t.segment_ties(df, cc)
    assert len(ties) == 1 and not ties[0].ok and ties[0].diff == -100
