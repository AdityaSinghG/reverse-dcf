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
