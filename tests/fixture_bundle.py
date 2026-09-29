"""TEST fixture: a synthetic company used to build and verify the workbook before real data exists.

Every number here is invented for testing and labelled `TEST FIXTURE`. None of it is Eternal data.
"""
import io

import pandas as pd

from src import data_loader as dl
from src.build_excel import ModelBundle
from src.monte_carlo import MCSpec

SRC, DATE = "TEST FIXTURE (synthetic - not real data)", "2025-01-01"

FY = {  # item: (FY2023, FY2024, FY2025), INR crore
    "revenue_from_operations": (400, 650, 1000),
    "total_expenses": (600, 850, 1100),
    "depreciation_amortisation": (40, 55, 70),
    "finance_costs": (10, 15, 20),
    "other_income": (30, 40, 55),
    "share_based_payment_expense": (12, 18, 25),
    "profit_before_tax": (-170, -160, -45),
    "tax_expense": (2, 3, 4),
    "net_profit": (-172, -163, -49),
    "cash_and_cash_equivalents": (120, 160, 200),
    "bank_balances_other": (100, 120, 150),
    "current_investments": (250, 280, 300),
    "non_current_investments": (100, 110, 120),
    "borrowings": (30, 35, 40),
    "lease_liabilities": (40, 50, 60),
    "non_controlling_interest": (5, 8, 10),
    "total_equity": (1100, 1250, 1400),
    "total_assets": (1300, 1450, 1600),
    "total_liabilities": (200, 200, 200),
    "cfo": (-90, -60, -10),
    "capex_ppe_intangibles": (40, 55, 70),
}
SEGMENTS_FY2025 = {"segment_revenue__food_delivery": 620, "segment_revenue__quick_commerce": 400,
                   "segment_revenue__eliminations": -20}
POINT = {  # single-date items (as at latest balance sheet)
    "unabsorbed_tax_losses": 450, "shares_outstanding_basic": 9.6,
    "shares_outstanding_diluted": 10.0, "esop_unamortised_cost": 15,
}
STATEMENT = {"revenue_from_operations": "IS", "cash_and_cash_equivalents": "BS", "cfo": "CF"}


def _rows():
    rows = []
    for item, vals in FY.items():
        for fy, v in zip((2023, 2024, 2025), vals):
            rows.append(f"testco,FY,{fy}-03-31,X,{item},{v},INR crore,{SRC},{DATE},TEST,")
    for item, v in SEGMENTS_FY2025.items():
        rows.append(f"testco,FY,2025-03-31,SEG,{item},{v},INR crore,{SRC},{DATE},TEST,")
    for item, v in POINT.items():
        unit = "crore shares" if item.startswith("shares") else "INR crore"
        rows.append(f"testco,POINT,2025-03-31,X,{item},{v},{unit},{SRC},{DATE},TEST,")
    return rows


def financials_csv_text() -> str:
    return ",".join(dl.LONG_COLUMNS) + "\n" + "\n".join(_rows()) + "\n"


def make_frame() -> pd.DataFrame:
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "fixture.csv"
    p.write_text(financials_csv_text(), encoding="utf-8")
    return dl.read_long_table(p, "testco")


def write_company_files(d):
    """Write a complete set of TEST data files for company 'testco'. Returns (config dict, peers path).
    Config = repo defaults + a testco block pointing at the files (readiness and loading tests)."""
    import pathlib
    d = pathlib.Path(d)
    (d / "fin.csv").write_text(financials_csv_text(), encoding="utf-8")
    hdr = ",".join(dl.LONG_COLUMNS)
    mk = [f"testco,POINT,2025-03-31,MKT,{i},{v},{u},{SRC},{DATE},TEST," for i, v, u in
          (("price", 320.0, "INR"), ("rf", 0.07, "ratio"), ("erp", 0.07, "ratio"), ("beta", 1.3, "x"), ("beta_se", 0.25, "x"))]
    (d / "mkt.csv").write_text(hdr + "\n" + "\n".join(mk) + "\n", encoding="utf-8")
    (d / "peers.csv").write_text(
        "name,levered_beta,debt,market_cap,tax_rate,source,source_date\n"
        f"TEST Peer A,1.2,50,500,0.25,{SRC},{DATE}\nTEST Peer B,1.0,20,400,0.25,{SRC},{DATE}\n", encoding="utf-8")
    cfg = dl.load_config()
    a = lambda v, ref: {"value": v, "ref": ref}
    cfg["companies"] = {"testco": {
        "name": "TESTCO Ltd (synthetic)", "currency": "INR", "first_fy": 2023, "last_fy": 2025, "fiscal_year_end_month": 3,
        "assumptions_as_of": "2025-01-01", "data_file": str(d / "fin.csv"), "market_data_file": str(d / "mkt.csv"),
        "segments": ["food_delivery", "quick_commerce"],
        "benchmarks": {"peers": [{"name": "TEST Peer A"}, {"name": "TEST Peer B"}]},
        "assumptions": {"terminal_growth": a(0.05, "A1"), "terminal_ebit_margin_range": a([0.12, 0.24], "A2"),
                        "base_case_terminal_ebit_margin": a(0.18, "A2"), "base_case_revenue_cagr": a(0.20, "A23"),
                        "sales_to_capital": a(2.5, "A3"), "cost_of_debt_pretax": a(0.09, "A4"),
                        "target_debt_weight": a(0.03, "A5"), "tax_rate_statutory": a(0.25, "A7"),
                        "tax_rate_terminal": a(0.25, "A7")},
        "monte_carlo": {"n_sims": 1000, "seed": 20250101, "correlation_cagr_margin": a(0.3, "A26"),
                        "distributions": {
                            "revenue_cagr": {"type": "triangular", "low": 0.12, "mode": 0.20, "high": 0.28},
                            "terminal_ebit_margin": {"type": "triangular", "low": 0.12, "mode": 0.18, "high": 0.24},
                            "wacc": {"type": "truncated_normal", "mean": 0.145, "sd": 0.01, "low": 0.12, "high": 0.17},
                            "terminal_growth": {"type": "triangular", "low": 0.04, "mode": 0.05, "high": 0.06}}},
    }}
    return cfg, d / "peers.csv"


def make_config() -> dict:
    return {
        "name": "TESTCO Ltd (synthetic)", "forecast_years": 10, "units": "INR crore",
        "share_unit": "crore shares", "segment_tolerance_pct": 0.5,
        "segments": ["food_delivery", "quick_commerce"],
        "bridge": {"asset_weights": {"cash_and_cash_equivalents": 1, "bank_balances_other": 1,
                                     "current_investments": 1, "non_current_investments": 0},
                   "claims": ["borrowings", "lease_liabilities", "non_controlling_interest"]},
        "discounting": {"mid_year": False},
        "forecast": {"growth_profile": "constant", "margin_ramp_years": None},
        "assumptions": {
            "terminal_growth": {"value": 0.05, "ref": "A1"},
            "sales_to_capital": {"value": 2.5, "ref": "A3"},
            "tax_rate_statutory": {"value": 0.25, "ref": "A7"},
            "tax_rate_terminal": {"value": 0.25, "ref": "A7"},
        },
    }


def make_bundle(mid_year: bool = False, profile: str = "constant", n_sims: int = 10_000) -> ModelBundle:
    cc = make_config()
    cc["discounting"]["mid_year"] = mid_year
    cc["forecast"]["growth_profile"] = profile
    a = lambda v, ref: {"value": v, "source": f"ASSUMPTION {ref}: TEST value", "date": DATE}
    peers = pd.DataFrame({
        "name": ["TEST Peer A", "TEST Peer B", "TEST Peer C"], "levered_beta": [1.2, 1.0, 1.4],
        "debt": [50.0, 20.0, 80.0], "market_cap": [500.0, 400.0, 600.0], "tax_rate": [0.25] * 3,
        "source": [SRC] * 3, "source_date": [DATE] * 3})
    spec = MCSpec(distributions={
        "revenue_cagr": {"type": "triangular", "low": 0.12, "mode": 0.20, "high": 0.28},
        "terminal_ebit_margin": {"type": "triangular", "low": 0.12, "mode": 0.18, "high": 0.24},
        "wacc": {"type": "truncated_normal", "mean": 0.145, "sd": 0.01, "low": 0.12, "high": 0.17},
        "terminal_growth": {"type": "triangular", "low": 0.04, "mode": 0.05, "high": 0.06}},
        n_sims=n_sims, seed=20250101, corr_cagr_margin=0.3)
    return ModelBundle(
        company_name="TESTCO Ltd (synthetic)", currency="INR", units="INR crore", is_test=True,
        cc=cc, df=make_frame(),
        market={"price": 320.0, "valuation_date": "2025-03-31", "source": SRC, "date": DATE},
        wacc_inputs={"rf": a(0.07, "B3"), "erp": a(0.07, "B4"), "beta": a(1.3, "A6"),
                     "beta_se": a(0.25, "A6"), "kd_pre_tax": a(0.09, "A4")},
        scenario={"revenue_cagr": a(0.20, "A23"), "terminal_ebit_margin": a(0.18, "A2")},
        peers=peers, mc_spec=spec,
        steps={"wacc": 0.01, "terminal_growth": 0.005, "revenue_cagr": 0.02, "terminal_ebit_margin": 0.02},
        tolerances={"balance_sheet": 0.005, "value_vs_python": 0.001, "price_reproduction": 0.001,
                    "grid_centre": 0.001, "esop_overlap_flag": 0.02, "beta_z": 1.96},
        run_date="2025-04-01")
