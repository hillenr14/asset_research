# EDGAR Strategy Module Plan

## Goal

Add a new standalone Python file to this project that, for a supplied stock ticker:

1. Downloads the last 10 years of annual (`10-K`) and quarterly (`10-Q`) SEC EDGAR reports into `data/`.
2. Reuses any already-downloaded local filing artifacts instead of downloading them again.
3. Extracts a usable set of key financial indicators from those filings.
4. Downloads historical price data from `yfinance`.
5. Generates buy/sell/hold signals from the financial indicators.
6. Runs a backtest to measure the strategy's performance.

This module will not be wired into `app.py` yet.

## Implementation Approach

## 1. Create a standalone module

Add a new file, tentatively `edgar_strategy.py`, with a small callable API and optional CLI entry point.

Proposed top-level flow:

1. Resolve ticker -> SEC CIK.
2. Pull SEC submissions metadata for the issuer.
3. Select `10-K` and `10-Q` filings within the trailing 10 years.
4. Download and cache filing artifacts under `data/`.
5. Pull SEC company facts for standardized XBRL values.
6. Build quarterly and annual financial datasets.
7. Compute derived indicators.
8. Pull daily price history from `yfinance`.
9. Convert filing-based indicators into point-in-time signals.
10. Backtest the signals with reporting lag and return a results bundle.

## 2. EDGAR data acquisition and local cache

Use SEC JSON endpoints for discovery and standardized facts, plus archive URLs for the filing documents themselves.

Data sources:

- `https://www.sec.gov/files/company_tickers.json`
  Purpose: map ticker to CIK.
- `https://data.sec.gov/submissions/CIK##########.json`
  Purpose: enumerate filing history and identify filing date, accession number, form, and primary document.
- `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
  Purpose: retrieve standardized reported facts derived from the issuer's XBRL filings.
- `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_no_dashes}/{primary_document}`
  Purpose: download the actual filing document.

Local storage layout:

- `data/10-K/{TICKER}/{filing_date}_{accession_no_no_dashes}.html`
- `data/10-K/{TICKER}/{filing_date}_{accession_no_no_dashes}.json`
- `data/10-Q/{TICKER}/{filing_date}_{accession_no_no_dashes}.html`
- `data/10-Q/{TICKER}/{filing_date}_{accession_no_no_dashes}.json`
- `data/companyfacts/{TICKER}.json`
- `data/analysis/{TICKER}_financials.csv`
- `data/analysis/{TICKER}_signals.csv`
- `data/analysis/{TICKER}_backtest.csv`
- `data/analysis/{TICKER}_summary.json`

Caching rules:

- If a filing HTML and metadata JSON already exist locally, skip that filing download.
- If company facts JSON exists, allow refresh via an explicit function flag, but default to reusing it.
- Keep metadata JSON beside each filing so later processing can trace back to the SEC accession and dates.

## 3. Financial fact extraction strategy

Do not try to fully parse raw inline XBRL from each HTML filing as the primary extraction path. That is materially more complex and brittle than using SEC standardized company facts.

Primary extraction path:

- Download filings to satisfy the archive requirement.
- Use SEC company facts JSON as the structured source for financial values that were reported in those filings.
- Align facts to filing/report dates and forms so the indicators are still based on the issuer's EDGAR-reported numbers.

Reason for this approach:

- More reliable across issuers than scraping filing HTML tables.
- Keeps dependencies light.
- Lets the module standardize around common US GAAP tags.

## 4. Financial datasets to build

Build two normalized DataFrames:

- Quarterly dataset keyed by fiscal period end / filing date.
- Annual dataset keyed by fiscal year end / filing date.

Core raw line items to collect when available:

- Revenue
- Net income
- Operating income
- Gross profit
- Operating cash flow
- Capital expenditures
- Free cash flow
- Cash and cash equivalents
- Total assets
- Total liabilities
- Current assets
- Current liabilities
- Long-term debt / total debt
- Stockholders' equity
- Shares outstanding
- EPS diluted

Tag mapping approach:

- Maintain a dictionary of preferred SEC XBRL tags and fallback tags for each concept.
- For each concept, choose the best available series by:
  1. Matching the right form (`10-Q` or `10-K`) and unit.
  2. Preferring higher coverage and fewer restatement duplicates.
  3. Deduplicating by fiscal period end, keeping the most recent filing if multiple rows map to the same period.

## 5. Derived indicators

From the normalized datasets, compute at least:

- Quarterly and annual revenue growth
- Net income growth
- EPS growth
- Gross margin
- Operating margin
- Net margin
- ROA
- ROE
- Current ratio
- Debt-to-equity
- Asset turnover
- Operating cash flow margin
- Free cash flow margin
- Free cash flow yield if market cap can be inferred
- Accrual proxy: `(net_income - operating_cash_flow) / total_assets`
- TTM revenue
- TTM EPS
- TTM free cash flow

Implementation detail:

- Use rolling 4-quarter windows for TTM fields.
- For balance-sheet ratios, align stock variables to the same reported quarter.

## 6. Signal design

The signal engine should be explicit, rule-based, and reproducible rather than a black-box model.

Use a composite fundamental score built from categories:

- Growth:
  positive revenue growth, EPS growth, TTM free cash flow growth
- Profitability:
  positive gross/operating/net margins, improving ROE/ROA
- Financial strength:
  current ratio above threshold, debt-to-equity below threshold
- Cash quality:
  operating cash flow positive, free cash flow positive, accruals not elevated
- Valuation overlay:
  price-to-sales or price-to-earnings relative to the issuer's own recent history when enough data exists

Initial scoring framework:

- Assign +1 / 0 / -1 to each signal component.
- Sum into a composite score.
- Map score to action:
  - `BUY` if score >= buy threshold
  - `HOLD` if score between thresholds
  - `SELL` if score <= sell threshold

Guardrails:

- Use only information available on or after the filing date.
- Apply a small publication lag, e.g. trade on the next market day after filing date.
- If critical metrics are missing, degrade gracefully instead of failing the entire run.

## 7. Price history and event alignment

Use `yfinance` daily adjusted price history for at least the same 10-year horizon.

Alignment rules:

- Join each filing-derived signal to the next available trading day.
- Forward-fill the most recent signal until the next filing event.
- Strategy position mapping:
  - `BUY` -> 1.0 exposure
  - `HOLD` -> 0.5 exposure
  - `SELL` -> 0.0 exposure

This produces a long-only, fundamental-timing strategy that is simple enough to test and explain.

## 8. Backtest design

Backtest assumptions for the first version:

- Daily revaluation using adjusted close returns.
- Position changes only when a new filing-based signal becomes active.
- No leverage, no shorting.
- Optional flat transaction cost per signal change, default small but configurable.

Performance outputs:

- Cumulative return
- CAGR
- Annualized volatility
- Sharpe ratio
- Max drawdown
- Win rate of signal periods
- Number of trades
- Benchmark comparison versus buy-and-hold of the same ticker

Artifacts to save:

- Daily backtest equity curve CSV
- Signal history CSV
- Summary metrics JSON

## 9. Public API shape

Planned functions:

- `run_edgar_strategy(ticker: str, years: int = 10) -> dict`
- `download_sec_filings(ticker: str, years: int = 10) -> list[dict]`
- `load_or_fetch_companyfacts(ticker: str, cik: str, refresh: bool = False) -> dict`
- `build_financial_datasets(companyfacts: dict) -> tuple[pd.DataFrame, pd.DataFrame]`
- `compute_financial_indicators(quarterly_df: pd.DataFrame, annual_df: pd.DataFrame) -> pd.DataFrame`
- `download_price_history(ticker: str, start_date: str) -> pd.DataFrame`
- `generate_signals(indicators_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame`
- `run_backtest(signal_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]`

## 10. Dependencies and constraints

Try to keep dependencies to the existing stack plus Python standard library.

Preferred libraries:

- `pandas`
- `numpy`
- `yfinance`
- `json`
- `pathlib`
- `urllib.request` or `requests` if we decide to add it

Avoid adding heavy SEC/XBRL parsing dependencies unless necessary.

## 11. Known limitations for version 1

- SEC company facts coverage varies by issuer and concept.
- Some concepts need tag fallbacks because filers use slightly different taxonomies.
- Financial institutions and insurers may need sector-specific metric logic.
- The generated signals will be heuristics, not investment advice.
- Backtest results will depend on data completeness and conservative event timing assumptions.

## 12. Validation plan

After implementation:

1. Run the module for a ticker with known filing history, likely `AAPL`.
2. Confirm it skips already-downloaded local filings.
3. Confirm it creates quarterly and annual indicator outputs.
4. Confirm it produces non-empty signals and a backtest summary.
5. Inspect edge cases where concepts are missing and ensure the module still completes.
