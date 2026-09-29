"""Turn a company's config + raw data files into a ModelBundle, or refuse with the exact list of gaps.

Nothing is defaulted: any missing data cell, unset assumption (A1-A5, A7, A23-A27) or unset date makes
`check_readiness` list it and `load_bundle` raise. No model numbers are produced from incomplete inputs.

Files (paths in config/companies.yaml):
  data_file          tidy long CSV of financials (same schema as data/templates/financials_template.csv)
  market_data_file   tidy long CSV, one POINT row each for: price, rf, erp, beta, beta_se
                     (period_end of the price row is the valuation date; rf/erp as decimals)
  data/raw/peers_beta_inputs.csv   name,levered_beta,debt,market_cap,tax_rate,source,source_date
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import data_loader as dl
from src import monte_carlo as mc
from src import wacc as wacc_mod
from src.build_excel import ModelBundle

ROOT = dl.ROOT
POINT_GROUPS = ("share_data", "tax", "esop")          # single-date items: latest value is enough
MARKET_ITEMS = ("price", "rf", "erp", "beta", "beta_se")
PEERS_FILE = "data/raw/peers_beta_inputs.csv"
REQUIRED_ASSUMPTIONS = {                                # config key -> (ref, description)
    "terminal_growth": ("A1", "terminal growth"),
    "terminal_ebit_margin_range": ("A2", "terminal EBIT margin range [low, high]"),
    "base_case_terminal_ebit_margin": ("A2", "base-case terminal EBIT margin"),
    "sales_to_capital": ("A3", "sales-to-capital"),
    "cost_of_debt_pretax": ("A4", "pre-tax cost of debt"),
    "target_debt_weight": ("A5", "target debt weight D/V"),
    "tax_rate_statutory": ("A7", "statutory tax rate"),
    "tax_rate_terminal": ("A7", "terminal tax rate"),
    "base_case_revenue_cagr": ("A23", "base-case revenue CAGR"),
}


@dataclass
class Readiness:
    company: str
    problems: dict = field(default_factory=dict)        # section -> list[str]

    def add(self, section: str, msg: str):
        self.problems.setdefault(section, []).append(msg)

    @property
    def ready(self) -> bool:
        return not self.problems

    @property
    def n(self) -> int:
        return sum(len(v) for v in self.problems.values())

    def text(self) -> str:
        if self.ready:
            return f"{self.company}: inputs complete."
        lines = [f"CANNOT BUILD {self.company}: {self.n} item(s) missing or invalid. Nothing was computed."]
        for section, msgs in self.problems.items():
            lines.append(f"\n[{section}]  ({len(msgs)})")
            lines += [f"  - {m}" for m in msgs]
        return "\n".join(lines)


def _path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _blank_assumption(v) -> bool:
    return v is None or (isinstance(v, (list, tuple)) and (len(v) != 2 or any(x is None for x in v)))


def check_readiness(key: str, cfg: dict | None = None) -> Readiness:
    cc = dl.company_config(key, cfg)
    rd = Readiness(key)

    # --- assumptions and dates -----------------------------------------------------------------
    for name, (ref, desc) in REQUIRED_ASSUMPTIONS.items():
        v = cc.get("assumptions", {}).get(name, {}).get("value")
        if _blank_assumption(v):
            rd.add("assumptions (notes/assumptions.md)", f"{ref}: {desc}  -> config assumptions.{name}.value")
    if not cc.get("assumptions_as_of"):
        rd.add("assumptions (notes/assumptions.md)", "assumptions_as_of: date you set the assumptions (ISO), recorded as their source date")
    if cc.get("last_fy") is None:
        rd.add("config", "last_fy: latest reported fiscal year in the data file")
    rng = cc.get("assumptions", {}).get("terminal_ebit_margin_range", {}).get("value")
    if not _blank_assumption(rng) and not rng[0] < rng[1]:
        rd.add("assumptions (notes/assumptions.md)", "A2: terminal_ebit_margin_range must be [low, high] with low < high")

    # --- Monte Carlo (A23-A27) ---------------------------------------------------------------------
    m = cc.get("monte_carlo") or {}
    if m.get("correlation_cagr_margin", {}).get("value") is None:
        rd.add("Monte Carlo (A23-A27)", "A26: correlation_cagr_margin.value")
    if m.get("seed") is None:
        rd.add("Monte Carlo (A23-A27)", "A27: seed")
    for name in mc.DRAWN:
        spec = (m.get("distributions") or {}).get(name)
        if not spec:
            rd.add("Monte Carlo (A23-A27)", f"distribution '{name}' missing")
            continue
        try:
            mc.make_dist(spec)
        except mc.MonteCarloError as e:
            ref = {"revenue_cagr": "A23", "terminal_ebit_margin": "A24", "wacc": "A25", "terminal_growth": "A25b"}[name]
            rd.add("Monte Carlo (A23-A27)", f"{ref} {name}: {e}")

    # --- files --------------------------------------------------------------------------------------
    fin_path, mkt_path = _path(cc["data_file"]), _path(cc["market_data_file"])
    peers_path = _path(PEERS_FILE)
    df = None
    if not fin_path.exists():
        n = f"all {len(template_rows(key, cc))} required rows" if cc.get("last_fy") else "one FY block (set last_fy in config to size it)"
        rd.add("data files", f"{cc['data_file']} not found. Blank template with {n}: "
                             f"data/templates/{key}_financials_blank.csv (run `python -m src.company_bundle {key} --write-templates`)")
    else:
        try:
            df = dl.read_long_table(fin_path, key)
        except (dl.DataError, ValueError) as e:
            rd.add("data files", f"{cc['data_file']}: {e}")
    if df is not None and cc.get("last_fy") is not None:
        gaps = _financial_gaps(df, cc)
        for g in gaps:
            rd.add("financials (data_file)", g)
        seg = dl.validate_segments(df, cc)
        last = seg[(seg["period_type"] == "FY") & (seg["period_end"].dt.year == cc["last_fy"])]
        if last.empty or last["status"].iloc[0] != "OK":
            st = "no consolidated revenue" if last.empty else last["status"].iloc[0]
            detail = "" if last.empty else f" (diff {last['diff_pct'].iloc[0]:.2f}% vs tolerance {last['tolerance_pct'].iloc[0]}%)"
            rd.add("financials (data_file)", f"segment revenue check for FY{cc['last_fy']}: {st}{detail}")
    if not mkt_path.exists():
        rd.add("data files", f"{cc['market_data_file']} not found: needs POINT rows for {', '.join(MARKET_ITEMS)} "
                             f"(template: data/templates/{key}_market_blank.csv)")
    else:
        try:
            mk = dl.read_long_table(mkt_path, key)
            have = set(mk[mk["value"].notna()]["line_item"])
            for it in MARKET_ITEMS:
                if it not in have:
                    rd.add("market data", f"{cc['market_data_file']}: row '{it}' has no value")
        except (dl.DataError, ValueError) as e:
            rd.add("data files", f"{cc['market_data_file']}: {e}")
    peers_cfg = [p["name"] for p in cc.get("benchmarks", {}).get("peers", [])]
    if not peers_path.exists():
        rd.add("data files", f"{PEERS_FILE} not found: columns {', '.join(wacc_mod.PEER_COLUMNS)} for {peers_cfg}")
    else:
        try:
            pe = wacc_mod.load_peer_inputs(peers_path)
            for name in peers_cfg:
                row = pe[pe["name"] == name]
                if row.empty or row[["levered_beta", "debt", "market_cap", "tax_rate"]].isna().any(axis=None):
                    rd.add("peers", f"{PEERS_FILE}: incomplete row for '{name}'")
        except (wacc_mod.WaccError, ValueError) as e:
            rd.add("data files", f"{PEERS_FILE}: {e}")
    return rd


def _financial_gaps(df: pd.DataFrame, cc: dict) -> list[str]:
    out = []
    per_year = {}
    point = []
    for group, items in cc["required_line_items"].items():
        (point if group in POINT_GROUPS else per_year.setdefault("fy", [])).extend(items)
    sub = {**cc, "required_line_items": {"fy": per_year.get("fy", [])}}
    gaps = dl.missing_report(df, sub)
    out += [f"{r.line_item} for FY{r.fy}" for r in gaps.itertuples()]
    for it in point:
        try:
            dl.latest_value(df, it)
        except dl.DataError:
            out.append(f"{it} (latest reported, any date)")
    have = set(df[df["value"].notna()]["line_item"])
    for it in list(cc["bridge"]["asset_weights"]) + list(cc["bridge"]["claims"]):
        if it not in have and not any(f"{it} for" in x for x in out):
            out.append(f"{it} (needed for the equity bridge)")
    units = set(df.loc[(df["line_item"] == "shares_outstanding_diluted") & df["value"].notna(), "unit"])
    if units and units != {cc.get("share_unit")}:
        out.append(f"shares_outstanding_diluted unit {sorted(units)} must be '{cc.get('share_unit')}'")
    return out


def template_rows(key: str, cc: dict | None = None) -> list[dict]:
    cc = cc or dl.company_config(key)
    first, last = cc.get("first_fy"), cc.get("last_fy") or cc.get("first_fy")
    rows = []
    mm = cc.get("fiscal_year_end_month", 3)
    end_day = {3: "31", 6: "30", 9: "30", 12: "31"}.get(mm, "28")
    fy_items = [i for g, its in cc["required_line_items"].items() if g not in POINT_GROUPS for i in its]
    pt_items = [i for g, its in cc["required_line_items"].items() if g in POINT_GROUPS for i in its]
    for fy in range(first, (last or first) + 1):
        for it in fy_items:
            rows.append(dict(company=key, period_type="FY", period_end=f"{fy}-{mm:02d}-{end_day}", statement="", line_item=it,
                             value="", unit=cc["units"], source="", source_date="", locator="", notes=""))
    for it in pt_items:
        unit = cc.get("share_unit") if it.startswith("shares_") else cc["units"]
        rows.append(dict(company=key, period_type="POINT", period_end="", statement="", line_item=it, value="",
                         unit=unit, source="", source_date="", locator="", notes="period_end = date of latest balance sheet"))
    for seg in cc.get("segments", []) + ["eliminations"]:
        rows.append(dict(company=key, period_type="FY", period_end=f"{last}-{mm:02d}-{end_day}", statement="SEG",
                         line_item=dl.SEGMENT_PREFIX + seg, value="", unit=cc["units"], source="", source_date="",
                         locator="", notes="segment note; eliminations negative"))
    return rows


def write_templates(key: str, out_dir: Path | None = None) -> list[Path]:
    cc = dl.company_config(key)
    out_dir = Path(out_dir) if out_dir else ROOT / "data" / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    fin = out_dir / f"{key}_financials_blank.csv"
    pd.DataFrame(template_rows(key, cc), columns=dl.LONG_COLUMNS).to_csv(fin, index=False)
    mk = out_dir / f"{key}_market_blank.csv"
    pd.DataFrame([dict(company=key, period_type="POINT", period_end="", statement="MKT", line_item=i, value="",
                       unit={"price": cc["currency"], "beta": "x", "beta_se": "x"}.get(i, "ratio"), source="",
                       source_date="", locator="", notes="period_end of 'price' = valuation date; rf/erp as decimals")
                  for i in MARKET_ITEMS], columns=dl.LONG_COLUMNS).to_csv(mk, index=False)
    pe = out_dir / "peers_beta_inputs_blank.csv"
    pd.DataFrame({"name": [p["name"] for p in cc.get("benchmarks", {}).get("peers", [])],
                  **{c: "" for c in wacc_mod.PEER_COLUMNS[1:]}}).to_csv(pe, index=False)
    return [fin, mk, pe]


def load_bundle(key: str, cfg: dict | None = None) -> ModelBundle:
    """Build the ModelBundle from real files. Raises with the full gap list if anything is missing."""
    rd = check_readiness(key, cfg)
    if not rd.ready:
        raise IncompleteInputs(rd)
    cc = dl.company_config(key, cfg)
    df = dl.read_long_table(_path(cc["data_file"]), key)
    mk = dl.read_long_table(_path(cc["market_data_file"]), key)

    def m(item):
        r = mk[(mk["line_item"] == item) & mk["value"].notna()].sort_values("period_end").iloc[-1]
        return {"value": float(r["value"]), "source": r["source"], "date": r["source_date"].date().isoformat(),
                "period_end": r["period_end"].date().isoformat()}

    price = m("price")
    asof = str(cc["assumptions_as_of"])
    a = cc["assumptions"]
    ass = lambda k: {"value": a[k]["value"], "source": f"ASSUMPTION {a[k].get('ref', '')}: notes/assumptions.md", "date": asof}
    peers = wacc_mod.load_peer_inputs(_path(PEERS_FILE))
    peers = peers[peers["name"].isin([p["name"] for p in cc["benchmarks"]["peers"]])].reset_index(drop=True)
    return ModelBundle(
        company_name=cc["name"], currency=cc["currency"], units=cc["units"], is_test=False, cc=cc, df=df,
        market={"price": price["value"], "valuation_date": price["period_end"], "source": price["source"], "date": price["date"]},
        wacc_inputs={"rf": m("rf"), "erp": m("erp"), "beta": m("beta"), "beta_se": m("beta_se"),
                     "kd_pre_tax": ass("cost_of_debt_pretax")},
        scenario={"revenue_cagr": ass("base_case_revenue_cagr"), "terminal_ebit_margin": ass("base_case_terminal_ebit_margin")},
        peers=peers, mc_spec=mc.spec_from_config(cc), steps=dict(cc["sensitivity_steps"]),
        tolerances=dict(cc["tolerances"]), run_date=dt.date.today().isoformat())


class IncompleteInputs(RuntimeError):
    def __init__(self, readiness: Readiness):
        super().__init__(readiness.text())
        self.readiness = readiness


if __name__ == "__main__":
    import sys
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    key = args[0] if args else "eternal"
    if "--write-templates" in sys.argv:
        for p in write_templates(key):
            print("wrote", p.relative_to(ROOT))
    else:
        r = check_readiness(key)
        print(r.text())
        raise SystemExit(0 if r.ready else 2)
