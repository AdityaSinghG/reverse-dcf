# Assumptions register

Every judgement call has an ID, a one-line rationale, and where it is enforced.
"Set" = fixed by me as a modelling choice (change it in `config/companies.yaml` if you disagree).
"OPEN" = needs YOUR decision; the code refuses to run without it (no silent defaults).

## A. Inputs that need your judgement (OPEN)

| ID | Assumption | Value | Rationale / how to decide |
|----|------------|-------|---------------------------|
| A1 | Terminal growth | OPEN | Must be < WACC (guarded) and should not exceed the INR risk-free rate (long-run nominal growth ceiling). |
| A2 | Terminal EBIT margin range | OPEN | Anchor to mature peers (sources C1-C4). Solved for in M3; a range is used for the frontier. |
| A3 | Sales-to-capital | OPEN | Damodaran industry tables (C4) as a start; Blinkit dark stores are capital-heavier than food delivery. |
| A4 | Pre-tax cost of debt | OPEN | Only matters if debt+leases are material; use borrowings-note rate, else rf + spread with the spread stated. |
| A5 | Target debt weight | OPEN | Market-value D/V with debt incl. leases at book, or an industry target. |
| A7 | Statutory tax rate and terminal tax rate | OPEN | Statutory rate incl. surcharge and cess (source B6). Terminal rate = statutory unless you argue otherwise. |

## B. Modelling choices (Set)

| ID | Choice | Value | Rationale |
|----|--------|-------|-----------|
| A6 | Beta window / index | 2 years, weekly, vs NIFTY 50, raw (no Blume) | Eternal listed Jul-2021, so history is short; 2y weekly is the common practitioner default, uses ~104 observations, and reflects the current (Blinkit-heavy) business rather than the loss-making listing era. Weekly avoids the thin-trading noise of daily and the ~24-point problem of monthly. `beta_window_grid()` shows 2/3/5y/full x weekly/monthly so the choice is visibly one of many; std error and R^2 are reported. |
| A11b | Which ESOP grants are in the diluted share count | Included: all options/RSUs GRANTED and OUTSTANDING (vested-unexercised and unvested). Excluded: the ungranted pool (reserved, not yet granted), awards already exercised (they are in basic shares), lapsed/cancelled awards. Trust-held shares: see check below. | Rule: a share is counted once, either in the share count or through the expense, never both. (1) Awards not yet granted have no share count effect; their future cost enters via the SBC expense inside future EBIT. (2) Awards already granted are claims on existing value, so they sit in the share count. (3) Known overlap, accepted as conservative: the forecast SBC expense in the first years still includes amortisation of the grant-date value of already-granted UNVESTED awards, which are also in the share count. Its size is the unamortised ESOP cost from the ESOP note (`esop_unamortised_cost`); the workbook Checks tab reports it as % of equity value, and if it exceeds 1-2% I will exclude unvested awards from the count and tell you. (4) Check to do on the filing: if an ESOP trust holds shares, confirm whether they are included in the issued share count while also being counted as outstanding options; if so, subtract trust-held shares (`esop_trust_shares`) from the basic count. Data lines to fill (optional but needed for this audit): `esop_options_outstanding_vested`, `esop_options_outstanding_unvested`, `esop_unamortised_cost`, `esop_pool_ungranted` (memo only, not counted), `esop_trust_shares`. Diluted total = basic + vested + unvested outstanding (treasury-method on exercise price if strikes are material, otherwise gross). |
| A12 | Beta cross-check | Peer median, unlevered (Hamada) then relevered at target D/E | Single-stock regression betas have wide confidence intervals; the peer table (Swiggy, DoorDash, Meituan) is a sanity check, not a replacement. If regression and peer betas differ by more than the regression's ~2 standard errors, I flag it and you decide which to use. Debt includes leases. |
| A8 | Cash bridge | Counted at 100%: cash and cash equivalents, other bank balances, current investments. Counted at 0%: non-current investments (pending note review) | Weights live in `bridge.asset_weights`. Non-current investments can include long-dated bank deposits (liquid in substance) AND strategic equity stakes (not liquid); I exclude them until you review the investments note and set a weight. The workbook lists every line and its weight. Treasury income (other income) is excluded from EBIT so cash is not counted twice. |
| A9 | Leases | Treated as debt everywhere | Under Ind AS 116 EBIT excludes lease interest (it sits in finance costs), so FCFF is pre-lease-financing. Therefore lease liabilities are (i) deducted in the equity bridge, and (ii) included in debt for WACC weights and beta unlevering. Mixing (EBIT after lease cost but leases also deducted) would double-count. Simplification: lease principal and new ROU assets are folded into reinvestment via sales-to-capital. |
| A10 | Tax and losses | Explicit NOL pool; statutory rate on taxable EBIT; converge to terminal rate | Loss years add to the pool; profit years use it first (`tax_schedule`). Opening pool = `unabsorbed_tax_losses` from the tax note. NOL still unused at year N is valued at tax_rate_terminal x balance, discounted from year N as a lump sum (conservative; it would really be used after N). Simplifications: no expiry of business losses (Indian law limits brought-forward business losses to 8 assessment years; unabsorbed depreciation is indefinite), no minimum alternate tax. Check the tax note for expiries; if material, reduce the opening pool. Terminal-year FCFF uses tax_rate_terminal on full EBIT. |
| A11 | Dilution and SBC | (i) SBC is a real expense: EBIT is taken from reported expenses, which already include ESOP cost, and is NEVER adjusted back. (ii) Per-share value uses diluted shares. | Company "adjusted EBITDA" excludes ESOP cost; I do not use it. To avoid double counting, diluted shares capture options already granted, while future grants are captured by the SBC expense inside future EBIT. `share_based_payment_expense` is a required line so the workbook shows SBC as % of revenue and verifies it sits inside `total_expenses`. Diluted count should come from the filing's Ind AS 33 diluted weighted-average or an ESOP-note treasury-method count; state which in `notes`. |
| A13 | Discounting | End-of-year flows (mid-year toggle exists, default off) | Simplest to audit and replicate in Excel. Mid-year multiplies value by (1+WACC)^0.5 exactly (tested). Base period is the last full FY (or LTM); valuation date is assumed equal to the base-period end. |
| A14 | Revenue growth shape | Constant CAGR over the 10 years; `linear_fade` available | Constant makes "implied revenue CAGR" a single, interpretable number. Terminal growth then applies from year 11. |
| A15 | Margin path | Linear from current operating EBIT margin to terminal margin, reached in year N unless `margin_ramp_years` says otherwise | Transparent; one parameter. Current margin is computed as revenue - (total expenses - finance costs), i.e. after D&A and SBC, before other income and finance cost. |
| A16 | Reinvestment | (Revenue_t - Revenue_{t-1}) / sales-to-capital; terminal reinvestment uses the same ratio at terminal growth | Terminal marginal ROIC = S2C x terminal margin x (1 - t) is reported so you can check growth is not value-destroying (ROIC below WACC) in perpetuity. |
| A17 | Equity bridge claims | Borrowings, lease liabilities, non-controlling interest, all at book value | NCI at book understates value if the subsidiary is worth more; note in memo if material. |
| A18 | Segment consistency | Loader flags segment revenue not summing to consolidated within 0.5% | `validate_segments`; segment rows use `segment_revenue__<name>` plus `segment_revenue__eliminations`. Statuses other than OK must be resolved before segment data is used. |

## B2. Reverse-solver conventions (Set, M3)

| ID | Choice | Rationale |
|----|--------|-----------|
| A19 | One parameter solved at a time (CAGR, terminal margin, WACC); everything else held at the inputs passed in, and every result prints the full held-fixed list | The implied number is only meaningful conditional on the rest; printing it stops it being quoted without its context. |
| A20 | Brackets are explicit and printed: CAGR (-5%, 60%), terminal margin (-5%, 60%), WACC (5%, 30%). Never widened automatically. | If the price is outside what the model can produce in the bracket the answer is "NO_SOLUTION_IN_RANGE", which is itself information (price implies something outside plausible territory). The one forced change is WACC's floor, raised to terminal growth + 0.01% because g >= WACC is invalid; it is disclosed in the result. |
| A21 | Grid scan (400 points) then brentq inside each sign change; several roots are all returned, none chosen | Loss pool + reinvestment can make value non-monotone in growth (test `test_multiple_roots_reported_from_loss_pool_case`). Unique-root assumptions would silently give a wrong "implied" number. |
| A22 | Frontier: for each CAGR, solve terminal margin. Points with no or several roots stay in the table with a status | Gaps in the curve are visible, not interpolated. |

## B3. Monte Carlo (M4). ALL distribution parameters are ASSUMPTIONS for you to set

Types/parameters live in `config/companies.yaml > defaults.monte_carlo.distributions` and are `null`
until you set them (the run refuses otherwise). They must come from history, peers or theory and must
NOT be tuned until the output looks right or the price is matched: that makes P(value >= price) circular.

| ID | Driver | Proposed type | What to anchor low / mode / high (or mean / sd) to |
|----|--------|---------------|-----------------------------------------------------|
| A23 | Revenue CAGR, 10y | triangular | worst / base / best 10-year growth: historical CAGR decay, mature-peer trajectories (C1-C3), TAM arguments |
| A24 | Terminal EBIT margin | triangular | peer and mature-platform margin range (C1-C4); bounded, right-skewed if you believe in scale economics |
| A25 | WACC | truncated normal | mean = CAPM WACC; sd from beta standard error (`BetaResult.std_err` x ERP) and ERP uncertainty; truncated to a plausible band |
| A25b | Terminal growth | triangular | upper bound at INR rf / long-run nominal GDP; g < WACC is enforced separately |
| A26 | Correlation of CAGR and terminal margin | +0.3 (documented default) | Scale platforms tend to have good and bad cases move together (operating leverage); the default is mild. The report ALWAYS shows results at 0 and at this value using identical random numbers. Set to 0 or negative if you think growth is bought with margin. Only this pair is correlated; WACC and g are independent of everything (a stated simplification). Gaussian copula, so marginals are unchanged. |
| A27 | Seed and n | seed 20250101 (arbitrary), n = 10,000 | Reproducibility; the seed is printed in every output. |

Mechanics you can defend: g >= WACC draws are rejected and redrawn (not truncated), and the count, share
and the before/after mean of WACC and g are reported so any bias is visible. Any draw that errors inside
the DCF is counted by error type and excluded, never dropped silently (`n_valid + n_errors = n`).
P(value >= price) is reported with its binomial standard error.

## B4. Workbook conventions (Set, M5)

| ID | Choice | Rationale |
|----|--------|-----------|
| A28 | Blue = input (Inputs tab only, each with source and date), black = formula, grey italic on grey = static Python output | Standard modelling colour code; enforced by `src/audit_excel.py`. |
| A29 | No numeric literal in a formula except 0 and 1; tolerances, z-values, timing offsets and axis steps are Inputs | Every number is visible and sourced. |
| A30 | Reverse solve values are static (Goal Seek cannot be created by openpyxl); the Reverse tab re-runs the full live DCF at each and checks it reproduces the price | The static number is auditable: it is proven by the live model, not trusted. |
| A31 | WACC x g grid is live closed-form when the growth profile is constant (years 1-10 FCFF do not depend on WACC or g); otherwise static. CAGR x margin grid is always static | A per-cell closed form exists only where the FCFF row is independent of the varied inputs. Centre cells are checked against the live model. |
| A32 | Excel is verified by LibreOffice headless recalculation and a cell-level comparison with Python (tolerance 0.1%) | If they differ, neither side is adjusted to fit; the differences are shown. |

## PRELIMINARY - ILLUSTRATIVE ASSUMPTIONS (tonight's first pass; NOT evidence-based)

Switch: `--preliminary` on the CLI or `preliminary.enabled: true` in `config/companies.yaml`. Every value lives in the
`preliminary:` block of the Eternal config and is a ROUND ILLUSTRATIVE number for you to edit or replace. Nothing here
is derived from history, peers or the price. With the mode off, the blank A1-A7 / A23-A27 gates apply again and the run
refuses. Every printed result, chart title, Excel sheet and the README carry the label.

| ID | Value | ILLUSTRATIVE rationale (one line, not evidence) |
|----|-------|--------------------------------------------------|
| A1 | terminal growth 5.0% | Round number below the assumed 7% risk-free rate; a placeholder for long-run nominal INR growth. |
| A2 | terminal EBIT margin range 5-12%, base 8% | Round band for a mature low-margin consumer platform on REPORTED revenue; the inventory-led basis makes any margin here highly uncertain. |
| A3 | sales-to-capital 3.0 | Round guess for a store/warehouse-heavy model; not taken from any industry table. |
| A4 | pre-tax cost of debt 9% | Round figure; immaterial while borrowings are assumed nil. |
| A5 | target debt weight 5% | Round figure standing in for lease liabilities as a share of capital; price-independent so WACC does not move with the placeholder price. |
| A6 | beta 1.2, s.e. 0.25 | ASSUMED input, not a regression (no price history supplied). |
| B3/B4 | risk-free 7.0%, ERP 7.0% | Round placeholders, not read from RBI/Damodaran. |
| A7 | statutory and terminal tax 25% | Round figure near the Indian statutory rate; not checked against the tax note. |
| A23 | base-case revenue CAGR 20% (Monte Carlo triangular 10/20/30%) | Round mid-point for 10 years of growth off reported LTM revenue; not fitted to history or to the price. |
| A24 | terminal margin triangular 4/8/12% | Same band as A2. |
| A25 | WACC ~ truncated normal, mean 15%, sd 1%, 12-18% | Centred near the CAPM result of the placeholder inputs. |
| A25b | terminal growth triangular 4/5/6% | Brackets A1. |
| A26 | correlation CAGR/margin +0.3 | Documented default from earlier in this file. |
| A27 | seed 20250101, n 10,000 | Arbitrary, for reproducibility. |

Non-assumption placeholders (also in the `preliminary:` block, each labelled as such in the workbook):
shares 921 crore (APPROXIMATE: paid-up capital INR 921 crore / face value INR 1, PDF p.25, no ESOP dilution);
price (you supply, approximate, never invented); opening tax-loss pool 0 (not in the filing; credits no shield);
unamortised ESOP cost 0 (not in the filing; the ESOP-overlap check is reported NOT ASSESSED); borrowings 0 (no
borrowings line on the balance sheet, PDF p.18; unconfirmed against the notes).
Bridge as at 30-Jun-2026 (PDF p.18, IMAGE_READ): cash + bank balances + current investments at 100%, non-current
investments at 0% (conservative placeholder), lease liabilities (non-current + current) as debt, NCI at book (negative).
Base period: LTM = FY26 + Q1FY27 - Q1FY26; LTM operating EBIT = PBT + finance costs - other income (SBC expensed).

## C. Not yet built (M2 scope decision needed, see chat)
The engine runs on consolidated revenue and margin. A per-segment revenue and margin build (food delivery,
Blinkit, Hyperpure, going-out) is not implemented; the segment data is currently validated and available
for a comparison table only.
