# TODO

## NEW

- New feature: add portfolio analysis with back test
  - Add new tab for portfolio analysis 

## Completed

- Added a global lookback control (`1w`, `1m`, `3m`, `6m`, `1y`, `2y`, `5y`, `10y`, `all`) above the tabs and updated the app to slice all analysis views from full cached histories in memory.
- Changed ticker CSV files to behave as full-history stores with tail-only refreshes, loaded into memory on app reload/startup, and suppressed crowded monthly/quarterly bar traces for long lookbacks.
- Renamed Portfolio Analysis to Holdings Analysis and replaced the raw imported table with a filtered/enriched holdings analysis table.
- Replaced the local `Investments.numbers` input with the live Holdings tab in the Investments Google Sheet.
- Removed the Analyze All button and now auto-analyze both modes on page load/refresh, with the first ticker selected by default.
- When a ticker is added, that ticker is immediately selected and analyzed.
- Updated the plus/minus controls to render as real buttons with emoji glyphs.
- Separated the ticker symbol button from the asset name, with the asset name rendered beside it in smaller text.
- Reduced the left pane width.
- Moved the analysis table to the left of the charts in the right pane.
- Removed Start Date and End Date from the analysis table display.
- Reduced bar widths in the charts to a maximum width of 10 days.
- Added chart grids and improved trace/bar colors for the dark theme.
- Simplified hover labels so the date appears once in the unified hover box.
- Reorganized the UI:
  - Two tabs on top of the page: one tab for Dividend Analysis (default tab) and one for Valuation Analysis.
  - Each tab has a left asset pane with add/remove/select controls and a main pane that shows the selected asset.
  - Clicking an asset analyzes it before displaying its output.
  - Historical analysis now uses a rolling two-year window and stores local cached price history, only fetching incremental updates when needed.
  - Removed the Dividend Combined Fundamentals & Metrics table.
- Split `app.py` into focused modules such as `data_provider.py`, `metrics.py`, `charts.py`, `config.py`, and a thinner Streamlit entrypoint.
- Introduced typed result objects for fundamentals, dividend analysis, and valuation analysis instead of passing loose dicts and DataFrames through the UI.
- Centralized Yahoo Finance parsing and normalization, including better handling for missing or inconsistent fields from `yfinance`.
- Improved error handling to distinguish between invalid ticker, unsupported analysis, missing statements, and transient provider failures.
- Improved the UI with stronger validation, clearer per-ticker status, and optional export of tables/charts.

## Deferred Improvements

- Replace rough metric approximations with more defensible calculations:
  - true trailing-four-quarter sums for EPS and revenue
  - configurable risk-free rate for Sharpe calculations
  - compounded return / total return metrics instead of arithmetic mean daily return times 252
- Add asset-type-aware behavior so ETFs/funds do not automatically use the same valuation logic as operating companies.
- Add automated tests for calculation logic and mocked Yahoo Finance payloads.
- Either promote useful notebook logic from `yfinance.ipynb` into reusable code or treat the notebook as archival research and reduce its maintenance footprint.
