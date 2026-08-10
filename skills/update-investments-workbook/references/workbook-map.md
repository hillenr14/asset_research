# Investments workbook map

## Tab roles

- `AI Update Instructions`: live, authoritative operating instructions. Read in full on every run.
- `Holdings_401k_import`: newest complete 401(k) holdings export.
- `history_401k_import`: cumulative, deduplicated 401(k) transaction history.
- `Holdings VG import`: Vanguard source data; it may contain holdings and history sections.
- `Holdings CS`: Charles Schwab holdings source.
- `History CS `: Charles Schwab history source. The visible name may have a trailing space; always use metadata.
- `Holdings`: current cross-institution positions and lot-level cost data.
- `Sold`: fully exited, called, matured, or redeemed positions.
- `Dividend Income`: monthly institution matrices plus annualized and per-share helpers.
- `Market Data`: live-yield formulas, fallbacks, effective yield, and source notes.
- Other reporting tabs, including `Monthly Summary` and `Change Plan`, may depend on these tables. Verify their affected formulas rather than rewriting them speculatively.

## Keys and classifications

- Reconcile current positions by `Ticker + Loc`.
- Valid `Loc` values are `401K`, `VG`, and `CS`.
- Valid `Type` values are `Stock`, `Income`, `Cash`, and `Growth`.
- Preserve separate Holdings rows for existing lots with different buy dates or costs.
- Aggregate those lots only when comparing quantity with a holdings export.

## History duplicate fingerprint

Build the fingerprint from every transaction column, excluding only a consistently blank leading spacer column. Before comparison:

1. Pad missing trailing columns with blanks.
2. Convert date values and Excel date serials to one canonical date form.
3. Parse numeric columns and round insignificant binary noise, normally to six decimal places.
4. Trim text, collapse repeated whitespace, and compare case-insensitively.
5. Keep the first occurrence in the workbook’s established order.

Do not use a partial key. Two same-day distributions or split orders can legitimately share ticker and amount.

## Income treatment

Include cash dividends, qualified dividends, money-market income, credit interest, and bond interest. Include capital-gain distributions only when the existing institution table already does so. Count reinvested income once using the income transaction; do not count the reinvestment purchase as income.

Use one row per calendar month and one column per asset in each institution’s Dividend Income matrix. Insert before annualized, total, or helper rows. Keep zeroes for months without eligible events when that is the existing convention.

## Structural edit cautions

- Inspect native table IDs and ranges immediately before structural edits.
- Insert new table rows before totals rows, copy a complete neighboring exemplar, then overwrite row-specific values.
- After inserting history rows, repair formulas whose absolute ranges shifted to preserve the former first data row; those formulas must include the newly inserted rows.
- After inserting an income month, verify totals include the new row.
- After adding an income asset column, extend Holdings and Sold lookup header/value ranges.
- After adding or removing Market Data rows, extend or shrink the native table range and all lookup bounds.
- Re-probe table metadata after all row and column changes.

## Expected final evidence

Capture and report:

- import market-value total;
- count of current exported positions;
- quantity mismatch list, which should be empty;
- extra current-Holdings list, which should be empty or documented;
- old duplicate count removed, incoming overlap count, unique new count, and final history count;
- newest month’s eligible income total and contributing assets;
- formula-error scan results;
- native-table and banding coverage;
- visual checks of Holdings, Sold, and the updated income area.
