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

## C. Not yet built (M2 scope decision needed, see chat)
The engine runs on consolidated revenue and margin. A per-segment revenue and margin build (food delivery,
Blinkit, Hyperpure, going-out) is not implemented; the segment data is currently validated and available
for a comparison table only.
