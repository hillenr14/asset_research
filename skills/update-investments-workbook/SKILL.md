---
name: update-investments-workbook
description: Update and reconcile the project’s recurring Investments Google Sheet from new brokerage holdings and transaction-history Excel exports. Use when Codex is asked to refresh 401(k), Vanguard, or Charles Schwab import tabs; merge history without duplicates; reconcile current and sold positions; update dividend-income matrices, annualized-income formulas, and Market Data; or verify the workbook after new investment data arrives.
---

# Update Investments Workbook

Use the live workbook as the authority for structure and formulas, and use the newest exports as the authority for current account data. Combine this skill with the Google Drive and Google Sheets skills.

## Required inputs

- Ground the exact Google Sheet by URL, ID, or an exact-title Drive search. The expected title is `Investments`; never select a similarly named file without verifying metadata.
- Identify the supplied holdings and history exports and the institution they represent.
- Read the complete `AI Update Instructions` tab at the start of every run. Those live instructions override this skill when they conflict.
- Read [references/workbook-map.md](references/workbook-map.md) for the known tab roles, reconciliation rules, and completion checklist.

## Workflow

1. Inspect spreadsheet metadata before cell reads. Record exact visible tab names, `sheetId` values, native tables, table ranges, totals rows, validation, and populated bounds.
2. Inspect each local export without modifying it. Run `scripts/analyze_exports.py --summary-only` to summarize its schema, as-of dates, positions, transactions, and duplicate fingerprints. Pass `--existing-history` when a local export of the live history tab is available.
3. Read bounded live ranges from the relevant import and derived tabs. Use `get_spreadsheet_cells` before writes that may affect formulas, validation, or formatting.
4. Build a reconciliation plan before editing:
   - replace a current holdings import with the newest complete export;
   - merge history by retaining every unique old row, adding only unique new rows, and removing pre-existing duplicates;
   - compare current positions by `Ticker + Loc` while preserving deliberate lot-level rows;
   - identify additions, full exits, redemptions, quantity changes, cash changes, new income events, and required Market Data entries.
5. Apply related edits in coherent batches. Preserve formulas, formats, dropdowns, totals rows, frozen headers, grouping, and native-table ranges. Use exact visible tab names.
6. Update all affected derived tabs required by `AI Update Instructions`, including Holdings, Sold, Dividend Income, and Market Data. Do not change unrelated institutions or reporting sections.
7. Re-read every changed range and run the verification checklist below. Use browser-based visual inspection when available.

## Non-negotiable rules

- Never discard a unique historical transaction.
- Define a duplicate using all transaction fields after normalizing dates, numeric precision, whitespace, casing, and trailing blank columns. Do not deduplicate using date, ticker, or amount alone.
- Treat a holdings export’s printed date as the holdings as-of date. Treat the newest transaction date as the history reconciliation date.
- Reconcile total position value, not share price, when applying the workbook’s materiality threshold.
- Treat fully exited, called, matured, or redeemed positions as Sold. Do not invent acquisition dates or cost basis.
- Count genuine income once. Exclude buys, sells, transfers, deposits, withdrawals, sweeps, and reinvestment purchases.
- Preserve lot-level rows when the workbook already tracks separate purchases. Aggregate lots only for comparison with the export.
- Remove temporary cash rows when they are absent from the newest complete holdings export and the account now reconciles without them.
- Add new income columns and months before helper or totals rows. Extend formulas and formatting through the exact new used range.
- Preserve live external-data formulas and their fallbacks. Record an unavailable fallback rather than inventing a yield.
- Re-read formula references after inserting or deleting rows. Structural edits can intentionally shift absolute references away from newly inserted history rows; repair them to cover the complete merged history.
- Never finish with unresolved `#REF!`, `#N/A`, `#VALUE!`, or `#DIV/0!` errors in affected ranges.

## Verification checklist

- The import tab matches the newest holdings export, including its printed as-of date and total market value.
- Aggregated `Ticker + Loc` quantities in Holdings equal the newest export, with no extra positions unless documented.
- The merged history contains every unique old row plus unique new rows and has zero duplicate fingerprints.
- Sold contains every newly exited security exactly once.
- Each new monthly income value ties to eligible source transactions; the monthly total reconciles.
- Annualized-income and per-share helper formulas reference the complete merged history and correct asset headers.
- Holdings and Sold lookup formulas include any added asset columns and the newest month.
- Market Data contains each current holding needed by formulas and no obsolete temporary-cash row.
- Native tables and banded ranges cover the final used rectangles; validation values remain intact.
- Changed tabs are visually legible and retain the workbook’s established green style.

## Completion summary

Report the source as-of dates, holdings added/removed/changed, Sold additions, income months/assets added, deduplication counts, assumptions or source limitations, reconciliation totals, formula/error results, and a direct link to the Google Sheet.
