# TODO

## Completed

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
