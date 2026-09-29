"""Load and validate company financials into tidy tables.

Design: every number enters the project through a tidy long CSV where each row
carries its own provenance (source, source_date, locator). The loader refuses
values without provenance and reports required line items that are missing,
so nothing is ever silently defaulted or invented.

Long schema (one row per company / period / line item):
    company, period_type (FY|Q|POINT), period_end (ISO date), statement,
    line_item, value, unit, source, source_date (ISO), locator, notes
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "companies.yaml"

LONG_COLUMNS = [
    "company", "period_type", "period_end", "statement", "line_item",
    "value", "unit", "source", "source_date", "locator", "notes",
]
PROVENANCE_COLUMNS = ["source", "source_date"]
PERIOD_TYPES = {"FY", "Q", "POINT"}


class DataError(ValueError):
    """Raised when input data violates the provenance / schema rules."""


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg or "companies" not in cfg:
        raise DataError(f"{path}: missing top-level 'companies' block")
    return cfg


def company_config(key: str, cfg: dict | None = None) -> dict:
    """Company block merged over `defaults` (company keys win)."""
    cfg = cfg or load_config()
    if key not in cfg["companies"]:
        raise DataError(f"Unknown company '{key}'. Known: {list(cfg['companies'])}")
    return {**cfg.get("defaults", {}), **cfg["companies"][key]}


def read_long_table(path: str | Path, company: str) -> pd.DataFrame:
    """Read a tidy long CSV, enforce schema and provenance, return typed frame.

    Rows with a blank `value` are kept (they document a known gap) but are
    excluded from `pivot_annual`. Rows WITH a value and no source/date raise.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See notes/SOURCES_NEEDED.md for what to download "
            "and data/templates/financials_template.csv for the format."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_cols = [c for c in LONG_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataError(f"{path.name}: missing columns {missing_cols}")

    df = df[df["company"].str.strip() == company].copy()
    for c in LONG_COLUMNS:
        df[c] = df[c].str.strip()

    bad_type = ~df["period_type"].isin(PERIOD_TYPES)
    if bad_type.any():
        raise DataError(f"{path.name}: period_type must be one of {PERIOD_TYPES}; "
                        f"bad rows: {df.index[bad_type].tolist()}")

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    if df["period_end"].isna().any():
        raise DataError(f"{path.name}: unparseable period_end at rows "
                        f"{df.index[df['period_end'].isna()].tolist()}")
    df["source_date"] = pd.to_datetime(df["source_date"].replace("", pd.NA), errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace("", pd.NA), errors="coerce")

    has_value = df["value"].notna()
    no_prov = has_value & (
        (df["source"] == "") | df["source_date"].isna()
    )
    if no_prov.any():
        raise DataError(
            f"{path.name}: {int(no_prov.sum())} value(s) lack source/source_date "
            f"(rule: no number without provenance). Rows: {df.index[no_prov].tolist()}"
        )
    dup = df.duplicated(["company", "period_type", "period_end", "line_item"], keep=False) & has_value
    if dup.any():
        raise DataError(f"{path.name}: duplicate (period, line_item) rows: "
                        f"{df.loc[dup, ['period_end', 'line_item']].values.tolist()}")
    return df.reset_index(drop=True)


def pivot_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Annual (FY) data as line_item x fiscal-year table. Only rows with values."""
    fy = df[(df["period_type"] == "FY") & df["value"].notna()].copy()
    fy["fy"] = fy["period_end"].dt.year  # FY ends in March: 31-Mar-2025 -> FY2025
    return fy.pivot(index="line_item", columns="fy", values="value").sort_index(axis=1)


def required_items(company_cfg: dict) -> list[str]:
    out: list[str] = []
    for items in company_cfg.get("required_line_items", {}).values():
        out.extend(items)
    return out


def missing_report(df: pd.DataFrame, company_cfg: dict) -> pd.DataFrame:
    """Required (line_item x FY) cells that have no value. Empty frame = complete."""
    first, last = company_cfg.get("first_fy"), company_cfg.get("last_fy")
    if first is None or last is None:
        raise DataError("Set first_fy and last_fy in config/companies.yaml to check completeness.")
    have = pivot_annual(df)
    rows = []
    for item in required_items(company_cfg):
        for fy in range(first, last + 1):
            ok = item in have.index and fy in have.columns and pd.notna(have.loc[item, fy])
            if not ok:
                rows.append({"line_item": item, "fy": fy})
    return pd.DataFrame(rows, columns=["line_item", "fy"])


SEGMENT_PREFIX = "segment_revenue__"
ELIMINATION_ITEM = SEGMENT_PREFIX + "eliminations"   # inter-segment eliminations / unallocated (signed)


def validate_segments(df: pd.DataFrame, company_cfg: dict,
                      tolerance_pct: float | None = None) -> pd.DataFrame:
    """Check sum(segment revenue) (+ eliminations line, if any) == consolidated revenue.

    Segment rows use line_item `segment_revenue__<segment>`; inter-segment eliminations use
    `segment_revenue__eliminations` (negative). One output row per (period_type, period_end)
    that has consolidated revenue or segment rows. status is one of:
      OK, MISMATCH, NO_CONSOLIDATED, NO_SEGMENT_DATA, UNKNOWN_SEGMENT
    Anything other than OK should be resolved before the segment view is used.
    """
    tol = company_cfg.get("segment_tolerance_pct", 0.5) if tolerance_pct is None else tolerance_pct
    known = set(company_cfg.get("segments", []))
    v = df[df["value"].notna()]
    seg = v[v["line_item"].str.startswith(SEGMENT_PREFIX)]
    cons = v[v["line_item"] == "revenue_from_operations"]
    keys = sorted(set(zip(seg["period_type"], seg["period_end"])) |
                  set(zip(cons["period_type"], cons["period_end"])))
    rows = []
    for ptype, pend in keys:
        s_rows = seg[(seg["period_type"] == ptype) & (seg["period_end"] == pend)]
        c_rows = cons[(cons["period_type"] == ptype) & (cons["period_end"] == pend)]
        consolidated = float(c_rows["value"].iloc[0]) if len(c_rows) else float("nan")
        seg_sum = float(s_rows["value"].sum()) if len(s_rows) else float("nan")
        names = {i[len(SEGMENT_PREFIX):] for i in s_rows["line_item"]} - {"eliminations"}
        unknown = sorted(names - known) if known else []
        if not len(s_rows):
            status, diff = "NO_SEGMENT_DATA", float("nan")
        elif not len(c_rows):
            status, diff = "NO_CONSOLIDATED", float("nan")
        else:
            diff = (seg_sum - consolidated) / consolidated * 100.0
            status = "OK" if abs(diff) <= tol else "MISMATCH"
        if unknown:
            status = "UNKNOWN_SEGMENT"
        rows.append({"period_type": ptype, "period_end": pend, "segment_sum": seg_sum,
                     "consolidated": consolidated, "diff_pct": diff,
                     "tolerance_pct": tol, "status": status,
                     "unknown_segments": ",".join(unknown)})
    return pd.DataFrame(rows, columns=["period_type", "period_end", "segment_sum", "consolidated",
                                       "diff_pct", "tolerance_pct", "status", "unknown_segments"])


def latest_value(df: pd.DataFrame, item: str, period_type: str | None = None):
    """Most recent populated value of a line item -> (value, period_end, source).
    Raises if none: never defaults."""
    v = df[(df["line_item"] == item) & df["value"].notna()]
    if period_type:
        v = v[v["period_type"] == period_type]
    if v.empty:
        raise DataError(f"no populated value for '{item}'")
    r = v.sort_values("period_end").iloc[-1]
    return float(r["value"]), r["period_end"], r["source"]


def ltm_sum(df: pd.DataFrame, item: str) -> float:
    """Sum of the latest four consecutive quarters (period_type Q) of a flow item."""
    q = df[(df["line_item"] == item) & (df["period_type"] == "Q") & df["value"].notna()]
    q = q.sort_values("period_end").tail(4)
    if len(q) < 4:
        raise DataError(f"LTM for '{item}' needs 4 quarters, have {len(q)}")
    gaps = q["period_end"].diff().dt.days.dropna()
    if ((gaps < 80) | (gaps > 100)).any():
        raise DataError(f"LTM quarters for '{item}' are not consecutive: {q['period_end'].dt.date.tolist()}")
    return float(q["value"].sum())


def build_processed(company: str, out_dir: Path | None = None) -> Path:
    """Validate raw data, write data/processed/<company>_annual.csv, return its path."""
    cc = company_config(company)
    df = read_long_table(cc["data_file"], company)
    annual = pivot_annual(df)
    out_dir = out_dir or ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{company}_annual.csv"
    annual.to_csv(out)
    return out


if __name__ == "__main__":
    import sys

    key = sys.argv[1] if len(sys.argv) > 1 else "eternal"
    cc = company_config(key)
    frame = read_long_table(cc["data_file"], key)
    print(f"{key}: {int(frame['value'].notna().sum())} populated rows")
    if cc.get("last_fy"):
        gaps = missing_report(frame, cc)
        print("COMPLETE" if gaps.empty else f"MISSING {len(gaps)} cells:\n{gaps.to_string(index=False)}")
    else:
        print("last_fy not set in config; completeness check skipped.")
