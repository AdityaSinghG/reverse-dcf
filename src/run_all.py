"""One command for the whole pipeline:  python -m src.run_all eternal

Order: readiness / loader validation -> WACC + forward DCF -> reverse solver -> sensitivities ->
Monte Carlo -> plots -> Excel build -> LibreOffice recalculation -> cell-level comparison with Python
-> workbook audit -> ONE summary.

If any input is missing or unset the command prints the exact list and stops BEFORE computing anything:
no model number is ever printed from incomplete inputs. `--fixture` runs the same pipeline on the
synthetic TEST data (every output is then labelled as test data and is not a result about any company).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openpyxl

from src import build_excel as be
from src import data_loader as dl
from src import monte_carlo as mc
from src import plots, reverse_solver as rs, sensitivity as sens

ROOT = dl.ROOT


def _hist_cagrs(df) -> dict:
    """Revenue CAGRs over every window ending in the last reported fiscal year."""
    fy = df[(df["line_item"] == "revenue_from_operations") & (df["period_type"] == "FY") & df["value"].notna()]
    s = fy.assign(y=fy["period_end"].dt.year).set_index("y")["value"].sort_index()
    out = {}
    last = s.index[-1]
    for first in s.index[:-1]:
        if s[first] > 0 and s[last] > 0:
            out[f"FY{first}-FY{last}"] = plots.cagr(s[first], s[last], last - first)
    return out


def run(key: str | None, fixture: bool = False) -> int:
    if fixture:
        from tests.fixture_bundle import make_bundle
        bundle, label, tag = make_bundle(), "TEST", "test"
        banner = "*** SYNTHETIC TEST DATA: no number below is a result about any company ***"
    else:
        from src.company_bundle import IncompleteInputs, load_bundle
        try:
            bundle = load_bundle(key)
        except IncompleteInputs as e:
            print(e.readiness.text())
            print("\nSTOPPED: fill the items above, then re-run. No model output was produced.")
            return 2
        label, tag, banner = key, key, None

    cc, df = bundle.cc, bundle.df
    lines = []
    out_dir = ROOT / "excel"
    if banner:
        lines.append(banner)

    # --- 1. loader validation --------------------------------------------------------------------
    seg = dl.validate_segments(df, cc)
    seg_last = seg[seg["period_type"] == "FY"].tail(1)
    lines.append(f"[1] DATA        {int(df['value'].notna().sum())} populated rows; segment check "
                 f"{seg_last['status'].iloc[0] if len(seg_last) else 'n/a'}; provenance enforced on every row")

    # --- 2-5. model -------------------------------------------------------------------------------
    py = be.python_model(bundle)
    inp, res, price, solved = py["inp"], py["res"], py["price"], py["solved"]
    lines.append(f"[2] WACC        {py['wacc']:.2%} (Ke {py['ke']:.2%}, Kd(1-t) {py['kd_at']:.2%}, D/V {py['wd']:.1%})")
    lines.append(f"[3] FORWARD     base case value/share {res.value_per_share:,.2f} vs price {price:,.2f} "
                 f"({res.value_per_share / price - 1:+.1%}); TV share of EV {res.tv_share_of_ev:.0%}")
    for p, r in solved.items():
        if r.status == rs.SOLVED:
            lines.append(f"[4] IMPLIED     {p}: {r.value:.2%}   (others held at base case)")
        elif r.status == rs.MULTIPLE:
            lines.append(f"[4] IMPLIED     {p}: MULTIPLE ROOTS " + ", ".join(f"{x:.2%}" for x in r.roots))
        else:
            lines.append(f"[4] IMPLIED     {p}: NO SOLUTION IN RANGE {r.bracket_used}")
    both = py["mc"]
    lines.append(f"[5] MONTE CARLO n={bundle.mc_spec.n_sims:,} seed={bundle.mc_spec.seed}: P(value>=price) "
                 f"{both['correlated'].prob_ge_price:.1%} (corr {bundle.mc_spec.corr_cagr_margin:+.1f}) / "
                 f"{both['uncorrelated'].prob_ge_price:.1%} (uncorrelated); median {both['correlated'].stats['p50']:,.2f}; "
                 f"g>=WACC redrawn {both['correlated'].n_rejected_g_ge_wacc:,}; DCF errors {both['correlated'].n_errors}")

    # --- 6. plots --------------------------------------------------------------------------------------
    src_note = ("TEST: synthetic inputs, not company data" if fixture else
                "Company filings and market data transcribed in data/raw (" +
                "; ".join(sorted(set(df['source'].dropna()))[:3]) + f"); model: reverse-dcf, run {bundle.run_date}")
    hist = _hist_cagrs(df)
    written = []
    P = lambda name: PLOTS(tag, name)
    if solved["revenue_cagr"].value is not None and hist:
        written.append(plots.plot_implied_vs_historical_cagr(
            solved["revenue_cagr"].value, {f"{k} (history)": v for k, v in hist.items()},
            title=f"Revenue CAGR the price implies vs history: {label}", source=src_note, path=P("implied_vs_historical_cagr")))
    fr = rs.implied_frontier(inp, price, np.linspace(0.0, 0.5, 26), n_scan=200)
    rng = cc.get("assumptions", {}).get("terminal_ebit_margin_range", {}).get("value")
    band = (rng[0], rng[1], "Assumed terminal margin range (A2)") if rng else None
    cband = (min(hist.values()), max(hist.values()), "Historical CAGR range") if hist else None
    if (fr["status"] == rs.SOLVED).any():
        written.append(plots.plot_frontier(
            fr, title=f"Growth and margin combinations that equal the price: {label}", source=src_note,
            margin_band=band, cagr_band=cband, points={"Base case": (inp.revenue_cagr, inp.terminal_ebit_margin)},
            path=P("implied_frontier")))
    written.append(plots.plot_mc_histogram(both["correlated"], uncorrelated=both["uncorrelated"],
                                           title=f"Simulated value per share vs market price: {label}",
                                           source=src_note, path=P("mc_histogram")))
    tor = sens.tornado(inp, sens.ranges_from_spec(bundle.mc_spec))
    written.append(plots.plot_tornado(tor, title=f"What moves value most: {label}", source=src_note, price=price,
                                      path=P("tornado")))
    lines.append(f"[6] PLOTS       {len(written)} written to plots/ ({', '.join(p.name for p in written)})")

    # --- 7-9. Excel ----------------------------------------------------------------------------------------
    out = out_dir / (f"reverse_dcf_{'TEST' if fixture else key}.xlsx")
    info, rec, cmp_, rep = be.build_and_check(bundle, out)
    bad = cmp_[~cmp_["ok"]]
    wb = openpyxl.load_workbook(rec, data_only=True)
    overall = be._cell(wb, info.cells["overall"])
    lines.append(f"[7] EXCEL       {out.name} recalculated by LibreOffice; {len(cmp_)} cells compared with Python, "
                 f"{len(bad)} beyond 0.1% (max rel. diff {cmp_['rel_diff'].max():.1e})")
    lines.append(f"[8] CHECKS TAB  {overall}")
    lines.append(f"[9] AUDIT       {'CLEAN' if not rep.findings else f'{len(rep.errors)} errors, {len(rep.warnings)} warnings'}")
    if len(bad):
        lines.append("    cell-level differences:\n" + bad.to_string())
    for f in rep.findings[:20]:
        lines.append(f"    {f.severity} {f.rule} {f.sheet}!{f.cell} {f.detail}")

    ok = not len(bad) and not rep.errors and overall != "FAILURES"
    lines.append("RESULT: " + ("ALL STAGES PASSED" if ok else "PROBLEMS FOUND (see above)") + (" [TEST DATA]" if fixture else ""))
    print("\n".join(lines))
    return 0 if ok else 1


def PLOTS(tag: str, name: str) -> Path:
    return plots.PLOTS_DIR / f"{tag}_{name}.png"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fx = "--fixture" in sys.argv
    if fx == bool(args):
        raise SystemExit("usage: python -m src.run_all <company>   |   python -m src.run_all --fixture")
    raise SystemExit(run(None if fx else args[0], fixture=fx))
