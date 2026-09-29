# Exact sources needed for Eternal Ltd (M1 deliverable)

I cannot fetch these for you without risking invented numbers. Download the files below into `data/raw/`
(git-ignored, never committed), then transcribe each number into `data/raw/eternal_financials.csv`
using the schema in `data/templates/financials_template.csv`. Every row needs `source`, `source_date`
and `locator` (page / note / table). Amounts in INR crore as reported.

## A. Company filings (primary sources)
| # | What | Where | Used for |
|---|------|-------|----------|
| A1 | Annual Reports, consolidated financial statements, **FY2022 to latest FY** (year ended 31 March) | eternal.com > Investor Relations, or BSE/NSE corporate filings (ETERNAL / BSE 543320) | Historicals: P&L, balance sheet, cash flow, share count, ESOP |
| A2 | Latest **quarterly results filing + investor shareholder letter** (most recent quarter, plus the 4 prior quarters) | Same; the shareholder letter has segment GOV, revenue, adj. EBITDA | Segment build, current margin, LTM run-rate |
| A3 | **Segment note** (Ind AS 108) in the annual report and quarterly results | Inside A1/A2 | Food delivery / quick commerce (Blinkit) / Hyperpure / going-out revenue and profit |
| A4 | Latest **balance sheet** (quarterly, consolidated) | A2 | Cash, bank balances, current + non-current investments, borrowings, lease liabilities, NCI for the EV-to-equity bridge |
| A5 | **Share count**: outstanding shares + ESOP options/RSUs outstanding (annual report ESOP note) | A1, NSE shareholding pattern | Diluted shares for per-share value |

## B. Market data
| # | What | Where | Used for |
|---|------|-------|----------|
| B1 | **Closing price** of ETERNAL on the valuation date (state the date) | NSE historical data (nseindia.com) | Price being reverse-solved |
| B2 | **Weekly (or monthly) closing prices** for ETERNAL and NIFTY 50, listing (Jul-2021) to valuation date | NSE historical data / NSE indices data | Beta regression (method documented in src/wacc.py) |
| B3 | **India 10-year G-Sec yield** on valuation date | RBI / FBIL / CCIL published yield | Risk-free rate |
| B4 | **India equity risk premium** (total ERP incl. country risk), latest January/July update | Damodaran, "Country Default Spreads and Risk Premiums" (pages.stern.nyu.edu/~adamodar) | ERP; record the dataset date |
| B5 | Corporate borrowing cost evidence (only if Eternal has meaningful debt) | Borrowings note in A1 (interest rate) | Pre-tax cost of debt; else labelled ASSUMPTION |
| B6 | **Statutory tax rate** (Income Tax Act s.115BAA, incl. surcharge and cess) | incometax.gov.in | Terminal tax rate; confirm Eternal's brought-forward losses in the tax note (A1) |

## C. Peer / history benchmarks (for the "is it believable" comparison)
| # | What | Where |
|---|------|-------|
| C1 | Swiggy (NSE: SWIGGY) latest annual report: revenue, adj. EBITDA, segment margins | swiggy.com IR / NSE |
| C2 | DoorDash 10-K (latest): revenue, adj. EBITDA, GOV | sec.gov EDGAR |
| C3 | Meituan annual report (latest): revenue, operating margin | hkexnews.hk |
| C4 | Damodaran industry margin / sales-to-capital tables (Global or India; Retail (Online), Restaurant/Dining) | pages.stern.nyu.edu/~adamodar (dataset date recorded) |

## D. Not available from public filings, needs YOUR judgement (see notes/assumptions.md)
Terminal growth, terminal EBIT margin range, sales-to-capital, target capital structure, beta method,
cost of debt if debt is immaterial. These are ASSUMPTIONS A1-A7 and are still blank in `config/companies.yaml`.

## Required line items
Listed in `config/companies.yaml` under `defaults.required_line_items`. Run
`python -m src.data_loader eternal` after filling data and setting `last_fy`; it prints every missing cell.

## Open questions I need you to confirm before M2
1. Valuation date (e.g. a specific closing date). Everything is priced off this.
2. Segment reporting: if the latest filing merged Hyperpure into another segment, tell me and I will
   update `segments:` accordingly rather than assume.
3. Peer set (Swiggy / DoorDash / Meituan proposed): OK?
