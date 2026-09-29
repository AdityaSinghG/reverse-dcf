"""Tie-outs on the extracted data. They REPORT differences; nothing is ever adjusted to make a tie work.

  balance sheet   total assets = total equity + total liabilities, at every balance-sheet date present
  segments        sum of segment revenue = consolidated revenue, every period that has segment rows
  cash flow       opening cash + net change + FX = closing cash; and closing cash = balance-sheet cash
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yaml

from src import data_loader as dl

ROOT = dl.ROOT


@dataclass
class Tie:
    name: str
    period: str
    lhs: float
    rhs: float
    tolerance: float                 # absolute, in the statement's unit

    @property
    def diff(self) -> float:
        return self.lhs - self.rhs

    @property
    def ok(self) -> bool:
        return abs(self.diff) <= self.tolerance

    def line(self) -> str:
        return (f"{'PASS' if self.ok else 'FAIL'}  {self.name} [{self.period}]: {self.lhs:,.1f} vs {self.rhs:,.1f} "
                f"(difference {self.diff:+,.1f}, tolerance {self.tolerance:g})")


def balance_sheet_ties(df: pd.DataFrame, abs_tol: float = 1.0) -> list[Tie]:
    """Assets vs equity + liabilities. abs_tol = 1 unit: statements are rounded to whole crore."""
    v = df[df["value"].notna()]
    out = []
    dates = sorted(set(v[v["line_item"] == "total_assets"]["period_end"]))
    for d in dates:
        g = lambda it: v[(v["line_item"] == it) & (v["period_end"] == d)]["value"]
        if all(len(g(i)) for i in ("total_assets", "total_equity", "total_liabilities")):
            out.append(Tie("total assets = total equity + total liabilities", str(d.date()),
                           float(g("total_assets").iloc[0]),
                           float(g("total_equity").iloc[0] + g("total_liabilities").iloc[0]), abs_tol))
    return out


def segment_ties(df: pd.DataFrame, cc: dict) -> list[Tie]:
    seg = dl.validate_segments(df, cc)
    out = []
    for r in seg.itertuples():
        if r.status in ("NO_SEGMENT_DATA", "NO_CONSOLIDATED"):
            continue
        tol = r.consolidated * r.tolerance_pct / 100.0
        out.append(Tie("sum of segment revenue = consolidated revenue" + (f" [{r.status}: {r.unknown_segments}]" if r.status == "UNKNOWN_SEGMENT" else ""),
                       f"{r.period_type} {r.period_end.date()}", r.segment_sum, r.consolidated, tol))
    return out


def cash_flow_ties(df: pd.DataFrame, image_reads_path: str) -> list[Tie]:
    """Reconcile the cash-flow statement and tie its closing cash to the balance sheet.
    Opening/FX/closing cash are read from the transcription file (they are not required model lines)."""
    ir = yaml.safe_load(open(ROOT / image_reads_path, encoding="utf-8"))
    grp = {g["name"]: g for g in ir.get("cf_foot_groups", [])}
    out = []
    g = grp["Closing cash (opening + net + FX)"]
    comp = g["components"]
    out.append(Tie("opening cash + net increase + FX = closing cash", "FY2026",
                   float(sum(comp.values())), float(g["total"]), 0.5))
    a, b, c = (grp[k]["total"] for k in ("Net cash from operating activities (A)", "Net cash from investing activities (B)",
                                          "Net cash from financing activities (C)"))
    out.append(Tie("A + B + C = net increase printed", "FY2026", float(a + b + c), float(grp["Net increase in cash (A+B+C)"]["total"]), 0.5))
    v = df[df["value"].notna() & (df["line_item"] == "cash_and_cash_equivalents") & (df["period_type"] == "FY")]
    if len(v):
        r = v.sort_values("period_end").iloc[-1]
        out.append(Tie("closing cash per cash flow = cash on the balance sheet", str(r["period_end"].date()),
                       float(g["total"]), float(r["value"]), 0.5))
    return out


def run_all_ties(df: pd.DataFrame, cc: dict, image_reads_path: str | None = None) -> list[Tie]:
    ties = balance_sheet_ties(df) + segment_ties(df, cc)
    if image_reads_path:
        ties += cash_flow_ties(df, image_reads_path)
    return ties


def report(ties: list[Tie]) -> str:
    fails = [t for t in ties if not t.ok]
    lines = [t.line() for t in ties]
    lines.append(f"TIE-OUTS: {len(ties) - len(fails)} of {len(ties)} pass; {len(fails)} failure(s)" +
                ("" if not fails else " (nothing was adjusted)"))
    return "\n".join(lines)
