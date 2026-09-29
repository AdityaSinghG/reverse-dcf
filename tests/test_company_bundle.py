"""Readiness (refusal with exact gaps) and loading from real-format files, using TEST data only."""
import copy
from pathlib import Path

import pandas as pd
import pytest

from src import build_excel as be, company_bundle as cb, data_loader as dl
from tests import fixture_bundle as fx


@pytest.fixture()
def files(tmp_path, monkeypatch):
    cfg, peers = fx.write_company_files(tmp_path)
    monkeypatch.setattr(cb, "PEERS_FILE", str(peers))
    return cfg, tmp_path


def problems(cfg):
    return cb.check_readiness("testco", cfg)


def flat(rd):
    return "\n".join(m for v in rd.problems.values() for m in v)


def test_complete_test_inputs_are_ready(files):
    cfg, _ = files
    rd = problems(cfg)
    assert rd.ready, rd.text()
    assert "inputs complete" in rd.text()


def test_load_bundle_builds_and_excel_matches_python(files, tmp_path):
    cfg, _ = files
    b = cb.load_bundle("testco", cfg)
    assert not b.is_test and b.market["valuation_date"] == "2025-03-31" and b.market["price"] == 320.0
    assert b.wacc_inputs["kd_pre_tax"]["source"].startswith("ASSUMPTION A4")
    assert len(b.peers) == 2 and b.mc_spec.seed == 20250101
    py = be.python_model(b)
    assert py["wd"] == pytest.approx(0.03) and py["we"] == pytest.approx(0.97)      # A5 target weight used
    if Path(be.SOFFICE).exists():
        info, rec, cmp_, rep = be.build_and_check(b, tmp_path / "x.xlsx")
        assert cmp_["ok"].all(), cmp_[~cmp_["ok"]].to_string()
        assert rep.findings == [], rep.text()


def test_missing_financial_cell_is_named_exactly(files):
    cfg, d = files
    df = pd.read_csv(d / "fin.csv", dtype=str, keep_default_na=False)
    df = df[~((df["line_item"] == "revenue_from_operations") & df["period_end"].str.startswith("2024"))]
    df.to_csv(d / "fin.csv", index=False)
    rd = problems(cfg)
    assert not rd.ready and "revenue_from_operations for FY2024" in flat(rd)
    with pytest.raises(cb.IncompleteInputs, match="revenue_from_operations for FY2024"):
        cb.load_bundle("testco", cfg)


def test_missing_point_item_and_bridge_item(files):
    cfg, d = files
    df = pd.read_csv(d / "fin.csv", dtype=str, keep_default_na=False)
    df = df[~df["line_item"].isin(["unabsorbed_tax_losses", "esop_unamortised_cost"])]
    df.to_csv(d / "fin.csv", index=False)
    text = flat(problems(cfg))
    assert "unabsorbed_tax_losses (latest reported" in text and "esop_unamortised_cost (latest reported" in text


def test_segment_mismatch_blocks(files):
    cfg, d = files
    df = pd.read_csv(d / "fin.csv", dtype=str, keep_default_na=False)
    m = df["line_item"] == "segment_revenue__quick_commerce"
    df.loc[m, "value"] = "300"
    df.to_csv(d / "fin.csv", index=False)
    assert "segment revenue check for FY2025: MISMATCH" in flat(problems(cfg))


def test_wrong_share_unit_blocks(files):
    cfg, d = files
    t = (d / "fin.csv").read_text(encoding="utf-8").replace("crore shares", "shares")
    (d / "fin.csv").write_text(t, encoding="utf-8")
    assert "shares_outstanding_diluted unit" in flat(problems(cfg))


@pytest.mark.parametrize("name,ref", [("terminal_growth", "A1"), ("terminal_ebit_margin_range", "A2"),
                                      ("sales_to_capital", "A3"), ("cost_of_debt_pretax", "A4"),
                                      ("target_debt_weight", "A5"), ("tax_rate_statutory", "A7"),
                                      ("base_case_revenue_cagr", "A23")])
def test_each_unset_assumption_is_listed(files, name, ref):
    cfg, _ = files
    cfg = copy.deepcopy(cfg)
    cfg["companies"]["testco"]["assumptions"][name]["value"] = None
    rd = problems(cfg)
    assert not rd.ready and f"{ref}:" in flat(rd) and name in flat(rd)
    assert len(rd.problems["assumptions (notes/assumptions.md)"]) == 1            # only that one


def test_zero_is_a_decision_not_a_gap(files):
    cfg, _ = files
    cfg = copy.deepcopy(cfg)
    cfg["companies"]["testco"]["assumptions"]["target_debt_weight"]["value"] = 0.0   # all-equity is a valid choice
    assert problems(cfg).ready


def test_unset_monte_carlo_parameters_listed(files):
    cfg, _ = files
    cfg = copy.deepcopy(cfg)
    m = cfg["companies"]["testco"]["monte_carlo"]
    m["distributions"]["wacc"]["sd"] = None
    m["correlation_cagr_margin"]["value"] = None
    text = flat(problems(cfg))
    assert "A25 wacc" in text and "A26" in text


def test_missing_market_row_peer_and_files(files):
    cfg, d = files
    t = "\n".join(l for l in (d / "mkt.csv").read_text(encoding="utf-8").splitlines() if ",beta_se," not in l)
    (d / "mkt.csv").write_text(t + "\n", encoding="utf-8")
    (d / "peers.csv").write_text("name,levered_beta,debt,market_cap,tax_rate,source,source_date\n"
                                 "TEST Peer A,1.2,50,500,0.25,TEST,2025-01-01\n", encoding="utf-8")
    text = flat(problems(cfg))
    assert "row 'beta_se' has no value" in text and "incomplete row for 'TEST Peer B'" in text
    (d / "mkt.csv").unlink()
    assert "mkt.csv not found" in flat(problems(cfg))


def test_value_without_source_is_reported_not_crashed(files):
    cfg, d = files
    t = (d / "mkt.csv").read_text(encoding="utf-8").replace("TEST FIXTURE (synthetic - not real data),2025-01-01", ",", 1)
    (d / "mkt.csv").write_text(t, encoding="utf-8")
    assert "provenance" in flat(problems(cfg))


def test_repo_config_refuses_and_lists_everything_until_you_fill_it():
    rd = cb.check_readiness("eternal")
    if rd.ready:
        pytest.skip("Eternal inputs are complete: refusal behaviour is covered by the TEST-data tests")
    text = rd.text()
    assert text.startswith("CANNOT BUILD eternal")
    for ref in ("A1:", "A2:", "A3:", "A4:", "A5:", "A7:", "A23:", "A23 revenue_cagr", "A24 terminal_ebit_margin",
                "A25 wacc", "A25b terminal_growth", "assumptions_as_of", "last_fy"):
        assert ref in text, ref
    with pytest.raises(cb.IncompleteInputs):
        cb.load_bundle("eternal")


def test_templates_list_every_required_row(tmp_path):
    cfg = dl.company_config("eternal")
    cfg = {**cfg, "last_fy": 2025}
    rows = cb.template_rows("eternal", cfg)
    items = {r["line_item"] for r in rows}
    for it in dl.required_items(cfg):
        assert it in items, it
    assert sum(r["line_item"] == "revenue_from_operations" for r in rows) == 2025 - cfg["first_fy"] + 1
    paths = cb.write_templates("eternal", tmp_path)
    assert [p.name for p in paths] == ["eternal_financials_blank.csv", "eternal_market_blank.csv", "peers_beta_inputs_blank.csv"]
    assert all(p.exists() for p in paths)
