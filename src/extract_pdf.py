"""Extract filed-statement numbers from a results PDF into the tidy financials CSV, with a verification sheet.

Rules (agreed with the user):
  * Text layer only. NO OCR. A page with no extractable text yields NOT FOUND for its lines.
  * Only the pages named in the extraction spec (filed statements), never summary tables in a letter.
  * Consolidated basis only. Where two filings present the same year, the latest filing wins and both
    versions are shown in the verification sheet (`other_filing_version`).
  * Every filled row carries source, source_date and a locator (PDF page + printed line + column).
  * A cell is filled only if its line is found with the expected number of columns; otherwise it is left
    blank and listed as NOT FOUND with the reason. Nothing is estimated.
  * Text-layer glitches are corrected only for one pattern (a '.' printed where a ',' thousands separator
    belongs, e.g. "1.396") and every such correction is flagged GLYPH-CORRECTED with the printed token shown.
    Sums of two printed lines (tax) are flagged DERIVED.

Usage:  python -m src.extract_pdf eternal config/extraction/eternal_q1fy27.yaml
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pdfplumber
import yaml

from src import company_bundle as cb
from src import data_loader as dl

ROOT = dl.ROOT
TOKEN = re.compile(r"\(\s*[\d][\d,\.]*\s*\)|\d[\d,\.]*|(?<=\s)-(?=\s|$)")
GLYPH = re.compile(r"^\d{1,3}\.\d{3}$")          # "1.396": '.' printed for the thousands separator


@dataclass
class Rec:
    line_item: str
    period: str
    period_end: str
    period_type: str
    status: str                       # FILLED | NOT FOUND
    value: float | None = None
    statement: str = ""
    pdf_page: int | None = None
    line_as_printed: str = ""
    column_header: str = ""
    printed_tokens: str = ""
    read_note: str = ""               # CLEAN | GLYPH-CORRECTED | DERIVED | reason for NOT FOUND
    other_filing_version: str = "n/a: only one filing supplied"
    document: str = ""
    source_date: str = ""


def parse_token(tok: str):
    """-> (value, note). Parentheses = negative. '.' as thousands separator is corrected and flagged."""
    t = tok.strip()
    neg = t.startswith("(")
    t = t.strip("() ").replace(" ", "")
    if t == "-":
        return 0.0, "CLEAN"
    note = "CLEAN"
    if GLYPH.match(t):
        t, note = t.replace(".", ""), "GLYPH-CORRECTED"
    else:
        t = t.replace(",", "")
    v = float(t)
    return (-v if neg else v), note


def numeric_tokens(line: str, after: int = 0) -> list[str]:
    return [m.group(0) for m in TOKEN.finditer(line[after:])]


def find_line(lines: list[str], pattern: str, start: int = 0):
    rx = re.compile(pattern, re.I)
    for i in range(start, len(lines)):
        m = rx.search(lines[i])
        if m:
            return i, m
    return None, None


def extract(spec: dict, pdf_path: Path) -> list[Rec]:
    recs: list[Rec] = []
    order = spec["column_order"]
    cols = [order.index(c) for c in spec["take_columns"]]
    per = spec["periods"]
    doc, sdate = spec["document"], spec["source_date"]
    with pdfplumber.open(pdf_path) as pdf:
        pages = {i + 1: (pg.extract_text() or "") for i, pg in enumerate(pdf.pages)}

    def mk(item, pcode, **kw):
        p = per[pcode]
        return Rec(line_item=item, period=pcode, period_end=p["period_end"], period_type=p["period_type"],
                   column_header=p["header"], document=doc, source_date=sdate, **kw)

    def read(page, label, start_idx=0, end_pat=None):
        text = pages.get(page, "")
        if not text.strip():
            return None, f"PDF p.{page} has no extractable text (no OCR performed)"
        lines = text.split("\n")
        i, m = find_line(lines, label, start_idx)
        if i is None:
            return None, f"line matching /{label}/ not found on PDF p.{page}"
        toks = numeric_tokens(lines[i], m.end())
        if len(toks) != len(order):
            return None, f"line '{lines[i][:70]}' has {len(toks)} numeric tokens, expected {len(order)}"
        return (lines[i], toks), None

    for t in spec["targets"]:
        got, err = read(t["page"], t["label"])
        for pc in spec["take_columns"]:
            if got is None:
                recs.append(mk(t["item"], pc, status="NOT FOUND", statement=t["statement"], pdf_page=t["page"], read_note=err))
                continue
            line, toks = got
            tok = toks[order.index(pc)]
            v, note = parse_token(tok)
            recs.append(mk(t["item"], pc, status="FILLED", value=v, statement=t["statement"], pdf_page=t["page"],
                           line_as_printed=line, printed_tokens=tok, read_note=note))
    for d in spec.get("derived", []):
        parts = [read(d["page"], p) for p in d["parts"]]
        for pc in spec["take_columns"]:
            if any(g is None for g, _ in parts):
                err = next(e for g, e in parts if g is None)
                recs.append(mk(d["item"], pc, status="NOT FOUND", statement=d["statement"], pdf_page=d["page"], read_note=err))
                continue
            toks = [g[1][order.index(pc)] for g, _ in parts]
            vals = [parse_token(t)[0] for t in toks]
            recs.append(mk(d["item"], pc, status="FILLED", value=sum(vals), statement=d["statement"], pdf_page=d["page"],
                           line_as_printed=" | ".join(g[0] for g, _ in parts), printed_tokens=" + ".join(toks),
                           read_note="DERIVED: " + d["note"]))
    seg = spec.get("segments")
    if seg:
        text = pages.get(seg["page"], "")
        lines = text.split("\n") if text.strip() else []
        s_i, _ = find_line(lines, seg["start"]) if lines else (None, None)
        e_i = None
        if s_i is not None:
            e_i, _ = find_line(lines, seg["end"], s_i + 1)
        for item, label in seg["rows"].items():
            for pc in spec["take_columns"]:
                if s_i is None or e_i is None:
                    recs.append(mk(item, pc, status="NOT FOUND", statement="SEG", pdf_page=seg["page"],
                                   read_note=f"segment block not located on PDF p.{seg['page']}"))
                    continue
                block = lines[s_i + 1:e_i]
                i, m = find_line(block, label)
                toks = numeric_tokens(block[i], m.end()) if i is not None else []
                if len(toks) != len(order):
                    recs.append(mk(item, pc, status="NOT FOUND", statement="SEG", pdf_page=seg["page"],
                                   read_note=f"segment line /{label}/ missing or has {len(toks)} numeric tokens"))
                    continue
                tok = toks[order.index(pc)]
                v, note = parse_token(tok)
                recs.append(mk(item, pc, status="FILLED", value=v, statement="SEG", pdf_page=seg["page"],
                               line_as_printed=block[i], printed_tokens=tok, read_note=note))
    return recs


def not_found_records(key: str, spec: dict, filled: list[Rec], last_fy: int) -> list[Rec]:
    """Every required (line x FY) and point item that no extraction filled, with the reason."""
    cc = {**dl.company_config(key), "last_fy": last_fy}
    have = {(r.line_item, r.period_end[:4], r.period_type) for r in filled if r.status == "FILLED"}
    out = []
    img_pages = {"balance_sheet": 18, "cash_flow": 19}
    for group, items in cc["required_line_items"].items():
        for it in items:
            if group in cb.POINT_GROUPS:
                out.append(Rec(line_item=it, period="latest", period_end="", period_type="POINT", status="NOT FOUND",
                               document=spec["document"],
                               read_note="not stated on any filed statement page of this document; needs annual report"))
                continue
            for fy in range(cc["first_fy"], last_fy + 1):
                if (it, str(fy), "FY") in have:
                    continue
                if fy == last_fy and group in img_pages:
                    why = (f"statement is on PDF p.{img_pages[group]} (letter annexure) as an image with no text layer; "
                           "not read (no OCR). Filed quarterly results do not include this statement")
                elif fy == last_fy:
                    why = "not found on the filed statement pages of this document"
                else:
                    why = "annual report for this year not supplied (extraction limited to this file)"
                out.append(Rec(line_item=it, period=f"FY{fy}", period_end=f"{fy}-{cc['fiscal_year_end_month']:02d}-31",
                               period_type="FY", status="NOT FOUND", document=spec["document"], read_note=why))
    return out


def read_checks(recs: list[Rec]) -> list[str]:
    """Footing of what was read (evidence the text layer was read correctly). NOT the tie-outs."""
    v = {(r.line_item, r.period): r.value for r in recs if r.status == "FILLED"}
    out = []
    for p in sorted({r.period for r in recs}):
        g = lambda k: v.get((k, p))
        if all(g(k) is not None for k in ("revenue_from_operations", "other_income", "total_expenses", "profit_before_tax")):
            d = g("revenue_from_operations") + g("other_income") - g("total_expenses") - g("profit_before_tax")
            out.append(f"{p}: revenue + other income - total expenses - PBT = {d:+.0f}")
        if all(g(k) is not None for k in ("profit_before_tax", "tax_expense", "net_profit")):
            out.append(f"{p}: PBT - tax - net profit = {g('profit_before_tax') - g('tax_expense') - g('net_profit'):+.0f}")
    return out


def write_outputs(key: str, spec: dict, recs: list[Rec], nf: list[Rec], out_csv: Path, out_dir: Path):
    cc = dl.company_config(key)
    rows = []
    filled = {(r.line_item, r.period_type, r.period_end): r for r in recs if r.status == "FILLED"}
    base = cb.template_rows(key, {**cc, "last_fy": max(int(r.period_end[:4]) for r in recs if r.period_type == "FY")})
    for row in base:
        if row["line_item"] == dl.ELIMINATION_ITEM:
            continue                                   # external-customer revenue basis: no eliminations line
        r = filled.pop((row["line_item"], row["period_type"], row["period_end"]), None)
        if r:
            row.update(value=f"{r.value:g}", source=spec["source"], source_date=spec["source_date"],
                       statement=r.statement, locator=f"{r.document} PDF p.{r.pdf_page}; '{r.line_as_printed[:60]}'; column '{r.column_header}'",
                       notes=r.read_note if r.read_note != "CLEAN" else "")
        else:
            row["notes"] = "NOT FOUND"
        rows.append(row)
    for r in filled.values():                          # extras (e.g. the latest quarter)
        rows.append(dict(company=key, period_type=r.period_type, period_end=r.period_end, statement=r.statement,
                         line_item=r.line_item, value=f"{r.value:g}", unit=spec["unit"], source=spec["source"],
                         source_date=spec["source_date"],
                         locator=f"{r.document} PDF p.{r.pdf_page}; '{r.line_as_printed[:60]}'; column '{r.column_header}'",
                         notes=r.read_note if r.read_note != "CLEAN" else ""))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=dl.LONG_COLUMNS).to_csv(out_csv, index=False)

    sheet = pd.DataFrame([vars(r) for r in recs + nf])
    sheet = sheet[["status", "line_item", "period", "period_end", "value", "pdf_page", "statement", "column_header",
                   "printed_tokens", "line_as_printed", "read_note", "other_filing_version", "document", "source_date"]]
    sheet = sheet.sort_values(["status", "period_end", "line_item"], ascending=[False, False, True])
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_dir / f"verification_{key}.csv", index=False)
    return sheet


def main(argv=None):
    argv = argv or sys.argv[1:]
    key, spec_path = argv[0], Path(argv[1])
    spec = yaml.safe_load(open(spec_path, encoding="utf-8"))
    pdf = ROOT / "data" / "raw" / spec["document"]
    recs = extract(spec, pdf)
    last_fy = max(int(r.period_end[:4]) for r in recs if r.period_type == "FY")
    nf = not_found_records(key, spec, recs, last_fy)
    cc = dl.company_config(key)
    sheet = write_outputs(key, spec, recs, nf, ROOT / cc["data_file"], ROOT / "data" / "processed")
    print(f"filled {sum(r.status == 'FILLED' for r in recs)}; not found {sum(r.status == 'NOT FOUND' for r in recs + nf)}")
    for line in read_checks(recs):
        print("read-check:", line)


if __name__ == "__main__":
    main()
