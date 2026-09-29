"""One command for the whole pipeline:  python -m src.run_all eternal [--preliminary --price P --price-date D]

Order: readiness -> tie-outs -> WACC + forward DCF -> reverse solver -> sensitivities -> Monte Carlo ->
plots -> Excel build -> LibreOffice recalculation -> cell-level comparison -> audit -> ONE summary.

FINAL mode (default): if any input is missing or unset the command prints the exact list and stops BEFORE
computing anything, so no model number is ever printed from incomplete inputs.

PRELIMINARY mode (`--preliminary` or preliminary.enabled in config): round illustrative assumptions are
overlaid (config `preliminary:` block). Every line printed, every chart title, and the Excel cover carry
"PRELIMINARY - ILLUSTRATIVE ASSUMPTIONS". The price is never invented: without `--price` the run does the
price-independent part only (base-case value per share and the two grids) and says what is waiting.

`--fixture` runs the same pipeline on synthetic TEST data (labelled as such).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import yaml

from src import build_excel as be
from src import data_loader as dl
from src import monte_carlo as mc
from src import plots, reverse_solver as rs, sensitivity as sens, tieouts

ROOT = dl.ROOT


def _hist_cagrs(df) -> dict:
    """Revenue CAGRs over every window ending in the last reported fiscal year (needs >= 2 fiscal years)."""
    fy = df[(df["line_item"] == "revenue_from_operations") & (df["period_type"] == "FY") & df["value"].notna()]
    s = fy.assign(y=fy["period_end"].dt.year).set_index("y")["value"].sort_index()
    out = {}
    for first in s.index[:-1]:
        if s[first] > 0 and s.iloc[-1] > 0:
            out[f"FY{first}-FY{s.index[-1]}"] = plots.cagr(s[first], s.iloc[-1], s.index[-1] - first)
    return out


def _grid_text(g: pd.DataFrame, row_fmt="{:.2%}", col_fmt="{:.2%}") -> str:
    g = g.copy()
    g.index = [row_fmt.format(x) for x in g.index]
    g.columns = [col_fmt.format(x) for x in g.columns]
    return g.to_string(float_format=lambda v: f"{v:,.1f}", na_rep="n/a")


def _tieouts(cc, df) -> list:
    spec_path = ROOT / "config" / "extraction" / f"{cc.get('ticker', '').lower()}_q1fy27.yaml"
    ir = None
    if spec_path.exists():
        ir = yaml.safe_load(open(spec_path, encoding="utf-8")).get("image_reads_file")
        if ir and not (ROOT / ir).exists():
            ir = None
    return tieouts.run_all_ties(df, cc, ir)


def run(key: str | None, fixture: bool = False, preliminary: bool = False, price: dict | None = None) -> int:
    lines: list[str] = []
    if fixture:
        from tests.fixture_bundle import make_bundle
        bundle, label, tag, pre = make_bundle(), "TEST", "test", ""
        lines.append("*** SYNTHETIC TEST DATA: no number below is a result about any company ***")
    else:
        from src.company_bundle import IncompleteInputs, effective_config, load_bundle, preliminary_on
        import src.company_bundle as cb
        cc0 = effective_config(key, None, preliminary)
        prelim = bool(cc0.get("_preliminary"))
        value_only = False
        try:
            bundle = load_bundle(key, preliminary=preliminary, price=price)
        except IncompleteInputs as e:
            only_price = prelim and set(e.readiness.problems) == {"market price (PRELIMINARY)"}
            if not only_price:
                print(e.readiness.text())
                print("\nSTOPPED: fill the items above, then re-run. No model output was produced.")
                return 2
            bundle = load_bundle(key, preliminary=True, price={"value": 1.0, "date": dt.date.today().isoformat()})
            value_only = True
        label, tag = key, key
        pre = f"[{bundle.label}] " if bundle.label else ""
        if bundle.label:
            lines.append(f"*** {bundle.label}: round illustrative inputs, NOT evidence-based; approximate price/share count ***")

    cc, df = bundle.cc, bundle.df
    P = lambda s: lines.append(pre + s)
    out_dir = ROOT / "excel"

    # --- 1. data + tie-outs -----------------------------------------------------------------------------
    seg = dl.validate_segments(df, cc)
    P(f"[1] DATA        {int(df['value'].notna().sum())} populated rows; provenance enforced on every row")
    try:
        ties = _tieouts(cc, df)
        P("[1b] TIE-OUTS   " + tieouts.report(ties).replace("\n", "\n" + pre + "               "))
    except Exception as e:                                        # a tie-out crash must be visible
        P(f"[1b] TIE-OUTS   COULD NOT RUN: {e}")

    # --- 2-3. model -------------------------------------------------------------------------------------
    if not fixture and value_only:
        py = be.python_model(bundle, price_outputs=False)
    else:
        py = be.python_model(bundle)
    inp, res, price_v = py["inp"], py["res"], py["price"]
    ltm = "LTM" if cc.get("forecast", {}).get("base") == "LTM" else "last FY"
    P(f"[2] BASE        {ltm} revenue {inp.base_revenue:,.0f}, operating EBIT margin {inp.base_ebit_margin:.2%} "
      f"(after D&A and SBC; before other income and finance costs), opening loss pool {inp.opening_nol:,.0f}")
    P(f"[2] WACC        {py['wacc']:.2%} (Ke {py['ke']:.2%}, Kd(1-t) {py['kd_at']:.2%}, D/V {py['wd']:.1%})")
    if fixture or not value_only:
        P(f"[3] FORWARD     base case value/share {res.value_per_share:,.2f} vs price {price_v:,.2f} "
          f"({res.value_per_share / price_v - 1:+.1%}); TV share of EV {res.tv_share_of_ev:.0%}")
    else:
        P(f"[3] FORWARD     base case value/share {res.value_per_share:,.2f} (EV {res.enterprise_value:,.0f}, equity {res.equity_value:,.0f}); "
          f"TV share of EV {res.tv_share_of_ev:.0%}")
    P("[3] GRID 1      value per share: WACC (rows) vs terminal growth (columns)")
    lines.append(_grid_text(py["grid_live"]))
    P("[3] GRID 2      value per share: revenue CAGR (rows) vs terminal EBIT margin (columns)")
    lines.append(_grid_text(py["grid_static"]))

    if not fixture and value_only:
        P("[4] WAITING     no price supplied: implied CAGR/margin/WACC, frontier chart, Monte Carlo P(value>=price), Excel "
          "and audit are skipped. Re-run with --price <INR> [--price-date YYYY-MM-DD].")
        lines.append(f"RESULT: VALUE-ONLY RUN COMPLETE ({bundle.label if bundle.label else ''})")
        print("\n".join(lines))
        return 0

    solved = py["solved"]
    for p, r in solved.items():
        if r.status == rs.SOLVED:
            P(f"[4] IMPLIED     {p}: {r.value:.2%}   (others held at base case)")
        elif r.status == rs.MULTIPLE:
            P(f"[4] IMPLIED     {p}: MULTIPLE ROOTS " + ", ".join(f"{x:.2%}" for x in r.roots))
        else:
            P(f"[4] IMPLIED     {p}: NO SOLUTION IN RANGE {r.bracket_used}")

    # --- Monte Carlo: reported only if it ran cleanly -----------------------------------------------------
    both = py["mc"]
    mc_ok = all(both[k].n_errors == 0 for k in both)
    if mc_ok:
        P(f"[5] MONTE CARLO n={bundle.mc_spec.n_sims:,} seed={bundle.mc_spec.seed}: P(value>=price) "
          f"{both['correlated'].prob_ge_price:.1%} (corr {bundle.mc_spec.corr_cagr_margin:+.1f}) / "
          f"{both['uncorrelated'].prob_ge_price:.1%} (uncorrelated); median {both['correlated'].stats['p50']:,.2f}; "
          f"g>=WACC redrawn {both['correlated'].n_rejected_g_ge_wacc:,}")
    else:
        P(f"[5] MONTE CARLO SKIPPED: {both['correlated'].n_errors} DCF errors in the draws (not reported as a result)")

    # --- plot: the frontier only in PRELIMINARY mode -----------------------------------------------------------
    ttl = f" ({bundle.label})" if bundle.label else ""
    src_note = ("TEST: synthetic inputs, not company data" if fixture else
                (f"{bundle.label}. Data: {cc['name']} Q1FY27 results filing (PDF p.18-19, 25-26); price approximate; "
                 f"assumptions illustrative; run {bundle.run_date}" if bundle.label else
                 "Company filings and market data transcribed in data/raw (" + "; ".join(sorted(set(df['source'].dropna()))[:3])
                 + f"); model: reverse-dcf, run {bundle.run_date}"))
    fr = rs.implied_frontier(inp, price_v, np.linspace(0.0, 0.5, 26), n_scan=200)
    written = []
    rng = cc.get("assumptions", {}).get("terminal_ebit_margin_range", {}).get("value")
    band = (rng[0], rng[1], "Assumed terminal margin range (A2, illustrative)" if bundle.label else "Assumed terminal margin range (A2)") if rng else None
    if (fr["status"] == rs.SOLVED).any():
        written.append(plots.plot_frontier(
            fr, title=f"Growth and margin combinations that equal the price: {label}{ttl}", source=src_note, margin_band=band,
            points={"Base case": (inp.revenue_cagr, inp.terminal_ebit_margin)}, path=plots.PLOTS_DIR / f"{tag}_implied_frontier.png"))
    if not bundle.label:                                            # final mode: the full chart set
        hist = _hist_cagrs(df)
        if solved["revenue_cagr"].value is not None and hist:
            written.append(plots.plot_implied_vs_historical_cagr(
                solved["revenue_cagr"].value, {f"{k} (history)": v for k, v in hist.items()},
                title=f"Revenue CAGR the price implies vs history: {label}", source=src_note,
                path=plots.PLOTS_DIR / f"{tag}_implied_vs_historical_cagr.png"))
        if mc_ok:
            written.append(plots.plot_mc_histogram(both["correlated"], uncorrelated=both["uncorrelated"],
                                                   title=f"Simulated value per share vs market price: {label}",
                                                   source=src_note, path=plots.PLOTS_DIR / f"{tag}_mc_histogram.png"))
        tor = sens.tornado(inp, sens.ranges_from_spec(bundle.mc_spec))
        written.append(plots.plot_tornado(tor, title=f"What moves value most: {label}", source=src_note, price=price_v,
                                          path=plots.PLOTS_DIR / f"{tag}_tornado.png"))
    P(f"[6] PLOTS       {len(written)} written to plots/ ({', '.join(p.name for p in written)})")

    # --- Excel: built and audited only if it runs cleanly -----------------------------------------------------
    ok = True
    suffix = "TEST" if fixture else (f"{key}_PRELIMINARY" if bundle.label else key)
    out = out_dir / f"reverse_dcf_{suffix}.xlsx"
    try:
        info, rec, cmp_, rep = be.build_and_check(bundle, out)
        bad = cmp_[~cmp_["ok"]]
        wb = openpyxl.load_workbook(rec, data_only=True)
        overall = be._cell(wb, info.cells["overall"])
        P(f"[7] EXCEL       {out.name} recalculated by LibreOffice; {len(cmp_)} cells compared with Python, "
          f"{len(bad)} beyond 0.1% (max rel. diff {cmp_['rel_diff'].max():.1e})")
        P(f"[8] CHECKS TAB  {overall}")
        P(f"[9] AUDIT       {'CLEAN' if not rep.findings else f'{len(rep.errors)} errors, {len(rep.warnings)} warnings'}")
        if len(bad):
            lines.append("    cell-level differences:\n" + bad.to_string())
        for f in rep.findings[:20]:
            lines.append(f"    {f.severity} {f.rule} {f.sheet}!{f.cell} {f.detail}")
        ok = not len(bad) and not rep.errors and overall != "FAILURES"
    except Exception as e:
        P(f"[7-9] EXCEL / AUDIT SKIPPED: did not run cleanly: {type(e).__name__}: {str(e)[:300]}")
        ok = False

    lines.append("RESULT: " + ("ALL STAGES PASSED" if ok else "PROBLEMS FOUND (see above)")
                 + (" [TEST DATA]" if fixture else (f" [{bundle.label}]" if bundle.label else "")))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    fx, pre = "--fixture" in argv, "--preliminary" in argv
    opts, args, i = {}, [], 0
    while i < len(argv):
        if argv[i] in ("--price", "--price-date"):
            opts[argv[i]] = argv[i + 1]
            i += 2
            continue
        if not argv[i].startswith("--"):
            args.append(argv[i])
        i += 1
    if fx == bool(args):
        raise SystemExit("usage: python -m src.run_all <company> [--preliminary --price P --price-date D]   |   python -m src.run_all --fixture")
    px = {"value": float(opts["--price"]), "date": opts.get("--price-date", dt.date.today().isoformat())} if "--price" in opts else None
    raise SystemExit(run(None if fx else args[0], fixture=fx, preliminary=pre, price=px))
