# Asset Research Dashboard

A Streamlit dashboard for researching dividend assets, equity valuations, and a live investment portfolio. Market prices and fundamentals come from Yahoo Finance through `yfinance`; portfolio holdings come from the `Holdings` worksheet in a Google Sheet.

## Features

- **Dividend analysis**
  - Price and adjusted-price history with dividend payments
  - Estimated dividend yield and payment frequency
  - Annualized return, volatility, Sharpe ratio, alpha, and beta metrics
  - Summary and per-ticker views
- **Valuation analysis**
  - Historical price-to-earnings and price-to-sales data
  - Quarterly EPS, revenue, and free-cash-flow context when available
  - Summary and per-ticker views
- **Holdings analysis**
  - Imports positions from Google Sheets and enriches them with current market data
  - Shows market value, estimated income, gains, and monthly income
  - Compares portfolio performance with SPY
  - Supports both recorded purchase dates and a hypothetical full-period view
- **Shared lookback control** from one week through full available history
- **Persistent ticker lists** in `config.json`
- **Local market-data caching** with automatic refresh after the latest NYSE close
- **Interactive Plotly charts** and selectable summary tables

## Requirements

- Python 3.10 or newer
- Internet access for Yahoo Finance and Google Sheets
- Read access to the configured `Investments` Google Sheet through a Google Cloud service account

## Installation

```bash
git clone https://github.com/hillenr14/asset_research.git
cd asset_research
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Google Sheets setup

The holdings page reads columns `A:G` from the `Holdings` worksheet in the Google Sheet configured in `portfolio_data.py`.

1. Create a Google Cloud service account and download its JSON key.
2. Share the Google Sheet with the service account's `client_email` as a viewer.
3. Configure credentials using one of these methods:

   - Add the service-account fields under `[gcp_service_account]` in `.streamlit/secrets.toml`.
   - Set `GOOGLE_SERVICE_ACCOUNT_JSON` to the complete JSON key value.
   - Set `GOOGLE_APPLICATION_CREDENTIALS` to the key file's path.
   - Set `google_application_credentials` in `.streamlit/secrets.toml` to the key file's path.

For example, a local secrets file can point to a key kept outside version control:

```toml
google_application_credentials = ".streamlit/google-service-account.json"
```

Both `.streamlit/secrets.toml` and `.streamlit/google-service-account.json` are ignored by Git. Never commit service-account credentials.

The first populated row returned from `A2:G` is treated as the header row and must include:

- `Ticker`
- `Type`
- `Loc`
- `Quantity`
- `Buy date`
- `Bought at`

Holdings are read until the first blank ticker after data begins. Cash rows use a fixed price and yield defined in `portfolio_data.py`; other rows are enriched using Yahoo Finance.

## Running the app

```bash
streamlit run app.py
```

The app refreshes holdings and warms the market-history cache when it starts, so Google Sheets credentials must be configured even if you initially plan to use only dividend or valuation analysis.

Use the controls at the top of the page to choose a lookback period and switch among:

- **Dividend Analysis** — add or remove tickers, open an asset, or view the group summary.
- **Valuation Analysis** — add or remove tickers, open an asset, or view the group summary.
- **Holdings Analysis** — review portfolio totals, income, performance, and supported asset details.

Ticker-list changes are saved automatically to `config.json`. The selected lookback determines the effective analysis start date; saved start-date values are refreshed to the rolling default by the application.

## Data and caching

Generated cache files live under `.cache/` and are excluded from Git:

- `.cache/price_history/` stores full per-ticker price and dividend histories.
- `.cache/ticker_snapshots/` stores ticker metadata and quarterly financial statements.

Cached data is reused during the trading day and refreshed when it no longer includes the latest expected NYSE session. Delete the relevant files under `.cache/` to force a clean download on the next run.

Data from Yahoo Finance may be delayed, incomplete, or unavailable for some symbols. The dashboard is a research tool and should not be treated as investment advice.

## Project structure

- `app.py` — Streamlit application, navigation, and UI state
- `charts.py` — Plotly chart construction
- `config.py` — lookback options and persisted ticker configuration
- `data_provider.py` — Yahoo Finance access and local caches
- `portfolio_data.py` — Google Sheets holdings import and portfolio calculations
- `metrics.py` — dividend, valuation, benchmark, and risk metrics
- `models.py` — shared data models
- `ui_helpers.py` — validation, formatting, and export helpers
- `errors.py` — application-specific errors and user-facing issue conversion
- `requirements.txt` — Python dependencies
- `config.json` — current dividend and valuation ticker lists
- `edgar_strategy.py` — separate SEC filing strategy research script
- `edgar_strategy_plan.md` — notes for the SEC filing strategy work
- `yfinance.ipynb` — exploratory Yahoo Finance notebook
