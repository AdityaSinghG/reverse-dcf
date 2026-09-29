import pandas as pd
import pytest

from src import data_loader as dl

HEADER = ",".join(dl.LONG_COLUMNS)


def write(tmp_path, rows):
    p = tmp_path / "fin.csv"
    p.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def test_config_loads_and_merges_defaults():
    cc = dl.company_config("eternal")
    assert cc["ticker"] == "ETERNAL" and cc["forecast_years"] == 10
    assert "revenue_from_operations" in dl.required_items(cc)


def test_unknown_company_raises():
    with pytest.raises(dl.DataError):
        dl.company_config("nope")


def test_value_without_provenance_rejected(tmp_path):
    p = write(tmp_path, ["eternal,FY,2025-03-31,IS,revenue_from_operations,100,INR crore,,,,"])
    with pytest.raises(dl.DataError, match="provenance"):
        dl.read_long_table(p, "eternal")


def test_blank_value_allowed_and_reported_missing(tmp_path):
    p = write(tmp_path, [
        "eternal,FY,2025-03-31,IS,revenue_from_operations,100,INR crore,TEST FIXTURE,2025-05-01,p1,",
        "eternal,FY,2025-03-31,IS,other_income,,INR crore,,,,gap",
    ])
    df = dl.read_long_table(p, "eternal")
    cfg = {"first_fy": 2025, "last_fy": 2025,
           "required_line_items": {"is": ["revenue_from_operations", "other_income"]}}
    gaps = dl.missing_report(df, cfg)
    assert gaps["line_item"].tolist() == ["other_income"]


def test_pivot_annual_fy_labels(tmp_path):
    p = write(tmp_path, [
        "eternal,FY,2024-03-31,IS,net_profit,10,INR crore,TEST FIXTURE,2024-05-01,,",
        "eternal,FY,2025-03-31,IS,net_profit,20,INR crore,TEST FIXTURE,2025-05-01,,",
    ])
    t = dl.pivot_annual(dl.read_long_table(p, "eternal"))
    assert t.loc["net_profit", 2025] == 20 and list(t.columns) == [2024, 2025]


def test_duplicate_rows_rejected(tmp_path):
    row = "eternal,FY,2025-03-31,IS,net_profit,20,INR crore,TEST FIXTURE,2025-05-01,,"
    with pytest.raises(dl.DataError, match="duplicate"):
        dl.read_long_table(write(tmp_path, [row, row]), "eternal")


# ---- segment validation, LTM, assembly -------------------------------------------------------
CFG = {"segments": ["food_delivery", "quick_commerce"], "segment_tolerance_pct": 0.5}


def _frame(tmp_path, rows):
    return dl.read_long_table(write(tmp_path, rows), "eternal")


def _r(item, val, end="2025-03-31", ptype="FY", stmt="IS"):
    return f"eternal,{ptype},{end},{stmt},{item},{val},INR crore,TEST FIXTURE,2025-05-01,,"


def test_segments_sum_ok_with_eliminations(tmp_path):
    df = _frame(tmp_path, [
        _r("revenue_from_operations", 1000),
        _r("segment_revenue__food_delivery", 600),
        _r("segment_revenue__quick_commerce", 410),
        _r("segment_revenue__eliminations", -10),
    ])
    out = dl.validate_segments(df, CFG)
    assert out["status"].tolist() == ["OK"]


def test_segments_mismatch_flagged(tmp_path):
    df = _frame(tmp_path, [
        _r("revenue_from_operations", 1000),
        _r("segment_revenue__food_delivery", 600),
        _r("segment_revenue__quick_commerce", 300),
    ])
    out = dl.validate_segments(df, CFG)
    assert out["status"].tolist() == ["MISMATCH"]
    assert out["diff_pct"].iloc[0] == pytest.approx(-10.0)


def test_segments_tolerance_boundary(tmp_path):
    df = _frame(tmp_path, [_r("revenue_from_operations", 1000),
                           _r("segment_revenue__food_delivery", 996)])
    assert dl.validate_segments(df, CFG)["status"].iloc[0] == "OK"          # -0.4% within 0.5%
    assert dl.validate_segments(df, CFG, tolerance_pct=0.1)["status"].iloc[0] == "MISMATCH"


def test_segments_missing_and_unknown(tmp_path):
    df = _frame(tmp_path, [
        _r("revenue_from_operations", 1000, end="2024-03-31"),
        _r("revenue_from_operations", 1000),
        _r("segment_revenue__mystery_seg", 1000),
    ])
    out = dl.validate_segments(df, CFG).set_index("period_end")["status"]
    assert out[pd.Timestamp("2024-03-31")] == "NO_SEGMENT_DATA"
    assert out[pd.Timestamp("2025-03-31")] == "UNKNOWN_SEGMENT"


def test_ltm_sum_and_gap_detection(tmp_path):
    qs = [("2024-06-30", 10), ("2024-09-30", 20), ("2024-12-31", 30), ("2025-03-31", 40)]
    df = _frame(tmp_path, [_r("revenue_from_operations", v, end=e, ptype="Q") for e, v in qs])
    assert dl.ltm_sum(df, "revenue_from_operations") == 100
    df2 = _frame(tmp_path, [_r("revenue_from_operations", v, end=e, ptype="Q")
                            for e, v in [qs[0], qs[1], qs[3], ("2025-06-30", 50)]])
    with pytest.raises(dl.DataError, match="consecutive"):
        dl.ltm_sum(df2, "revenue_from_operations")


def test_latest_value_raises_when_absent(tmp_path):
    df = _frame(tmp_path, [_r("net_profit", 5)])
    assert dl.latest_value(df, "net_profit")[0] == 5
    with pytest.raises(dl.DataError):
        dl.latest_value(df, "borrowings")


def test_inputs_from_data_fixture(tmp_path):
    from src import dcf
    rows = [
        _r("revenue_from_operations", 1000), _r("total_expenses", 1100), _r("finance_costs", 20),
        _r("cash_and_cash_equivalents", 100, stmt="BS"), _r("current_investments", 50, stmt="BS"),
        _r("borrowings", 30, stmt="BS"), _r("lease_liabilities", 20, stmt="BS"),
        _r("non_controlling_interest", 5, stmt="BS"), _r("unabsorbed_tax_losses", 300, stmt="TAX"),
        _r("shares_outstanding_diluted", 10, stmt="SH").replace("INR crore", "crore shares"),
    ]
    df = _frame(tmp_path, rows)
    cc = {"forecast_years": 10, "share_unit": "crore shares",
          "bridge": {"asset_weights": {"cash_and_cash_equivalents": 1, "current_investments": 1},
                     "claims": ["borrowings", "lease_liabilities", "non_controlling_interest"]},
          "assumptions": {"terminal_growth": {"value": None}, "sales_to_capital": {"value": None},
                          "tax_rate_statutory": {"value": None}, "tax_rate_terminal": {"value": None}}}
    with pytest.raises(dl.DataError, match="not set"):
        dcf.inputs_from_data(df, cc, wacc=0.12, revenue_cagr=0.1, terminal_ebit_margin=0.15)
    inp = dcf.inputs_from_data(df, cc, wacc=0.12, revenue_cagr=0.1, terminal_ebit_margin=0.15,
                               terminal_growth=0.05, sales_to_capital=2.0,
                               tax_rate_statutory=0.25, tax_rate_terminal=0.25)
    assert inp.base_ebit_margin == pytest.approx(-0.08)
    assert inp.opening_nol == 300 and inp.diluted_shares == 10
    assert sum(inp.bridge.values()) == pytest.approx(100 + 50 - 30 - 20 - 5)
    assert dcf.run_dcf(inp).enterprise_value > 0          # runs end to end


def test_share_unit_mismatch_rejected(tmp_path):
    from src import dcf
    df = _frame(tmp_path, [
        _r("revenue_from_operations", 1000), _r("total_expenses", 900), _r("finance_costs", 0),
        _r("cash_and_cash_equivalents", 1, stmt="BS"), _r("unabsorbed_tax_losses", 0),
        _r("shares_outstanding_diluted", 10)])   # unit left as "INR crore" -> wrong
    cc = {"share_unit": "crore shares",
          "bridge": {"asset_weights": {"cash_and_cash_equivalents": 1}, "claims": []}}
    with pytest.raises(dl.DataError, match="unit"):
        dcf.inputs_from_data(df, cc, wacc=0.12, revenue_cagr=0.1, terminal_ebit_margin=0.15,
                             terminal_growth=0.05, sales_to_capital=2.0,
                             tax_rate_statutory=0.25, tax_rate_terminal=0.25)
