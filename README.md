# reverse-dcf

> **IN PROGRESS. PRELIMINARY - ILLUSTRATIVE ASSUMPTIONS.** Nothing here is a finished result or investment advice.
> The assumptions behind every number below are round placeholders I chose, not evidence.

## Thesis (the question)
What revenue growth, terminal EBIT margin and discount rate does the market price of Eternal Ltd (NSE: ETERNAL,
consolidated, March year-end) imply, and is that believable against the company's own history and peers?
A reverse DCF fixes the price and solves for one driver at a time, so the conclusion is about what has to be true, not
about a point estimate of value.

## Preliminary result (one line)
Under illustrative assumptions (20% revenue CAGR, 8% terminal EBIT margin, 15.0% WACC, 5% terminal growth) the base-case
value is about INR 32 per share on LTM revenue of INR 67,408 crore and an LTM operating EBIT margin of -0.2%;
the implied CAGR/margin/WACC and the frontier chart are pending the market price (not supplied, not invented).
PRELIMINARY - ILLUSTRATIVE ASSUMPTIONS.

## Revenue basis used
Reported consolidated revenue from operations, as filed. It is NOT like-for-like across time: from Q1FY26 the quick
commerce (Blinkit) business began moving from a marketplace (commission) model to a partly inventory-led model
(note 4, PDF p.26), which books gross sales as revenue and reduced Hyperpure B2B revenue. LTM (Jul-2025 to Jun-2026) is
therefore mostly post-transition, but any comparison with earlier years or with marketplace-model peers is
distorted, and a terminal margin applied to inventory-led revenue must be read accordingly.

## Limitations (read before using any number)
- **Single filing.** Data come only from `eternal_Q1FY27_results.pdf` (filed statements PDF p.25-26; balance sheet and
  cash flow read from page images, p.18-19, flagged IMAGE_READ). No FY2022-FY2025 history, no share count, no tax-loss
  pool, no ESOP note, no peer data.
- **Illustrative assumptions.** Growth, margin, tax, sales-to-capital, beta, risk-free rate, ERP and every Monte Carlo
  distribution are round placeholders (see `notes/assumptions.md`, section PRELIMINARY).
- **Approximate inputs.** Share count is paid-up capital / face value (no dilution); price is user-supplied and approximate;
  borrowings assumed nil; non-current investments excluded from the bridge (conservative placeholder).
- **Tie-outs.** Balance sheet (2 dates), segment sum (3 periods) and cash-flow reconciliation all tie on the extracted cells.

## How to run
    python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
    .venv\Scripts\python -m pytest
    .venv\Scripts\python -m src.run_all eternal --preliminary                        # value-only, no price needed
    .venv\Scripts\python -m src.run_all eternal --preliminary --price <INR> --price-date YYYY-MM-DD
    .venv\Scripts\python -m src.run_all eternal                                      # FINAL mode: refuses until inputs are real
Excel needs LibreOffice for the recalculation check (path in `src/build_excel.py`). Built workbooks are never committed.
Structure: `src/` (loader, WACC, DCF, solver, sensitivity, Monte Carlo, plots, Excel build, audit, tie-outs, PDF extractor),
`config/companies.yaml`, `notes/assumptions.md`, `tests/`.
