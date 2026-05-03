from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from numbers_parser import Document

from data_provider import get_full_price_history, get_ticker_snapshot
from errors import MissingDataError, ProviderError


PORTFOLIO_DOCUMENT_PATH = (
    Path.home() / "Library/Mobile Documents/com~apple~Numbers/Documents/Investments.numbers"
)
PORTFOLIO_SHEET_NAME = "Holdings"
PORTFOLIO_TABLE_NAME = "Holdings"
PORTFOLIO_CACHE_TTL_SECONDS = 86400
PORTFOLIO_SOURCE_COLUMNS = ["Ticker", "Type", "Loc", "Quantity", "Buy date", "Bought at"]
CASH_PRICE = 1.0
CASH_YIELD = 0.035


def _normalize_portfolio_value(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _load_holdings_table_from_numbers(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MissingDataError(f"Portfolio file not found: {path}")

    try:
        document = Document(str(path))
    except Exception as exc:
        raise ProviderError(f"Could not open portfolio file at {path}.", str(exc)) from exc

    for sheet in document.sheets:
        if sheet.name != PORTFOLIO_SHEET_NAME:
            continue
        for table in sheet.tables:
            if table.name != PORTFOLIO_TABLE_NAME:
                continue
            rows = table.rows(values_only=True)
            if not rows:
                raise MissingDataError("The Holdings table is empty.")
            headers = [str(header).strip() for header in rows[0]]
            data_rows = rows[1:]
            frame = pd.DataFrame(data_rows, columns=headers)
            frame = frame.map(_normalize_portfolio_value)
            for column in frame.columns:
                if frame[column].dtype == "object":
                    frame[column] = frame[column].where(frame[column].notna(), "").astype(str)
            missing_columns = [column for column in PORTFOLIO_SOURCE_COLUMNS if column not in frame.columns]
            if missing_columns:
                raise MissingDataError(
                    f"Holdings table is missing required columns: {', '.join(missing_columns)}"
                )

            filtered = frame[PORTFOLIO_SOURCE_COLUMNS].copy()
            filtered = filtered.rename(columns={"Buy date": "Buy Date"})
            filtered["Ticker"] = filtered["Ticker"].astype(str).str.strip().str.upper()
            filtered = filtered[filtered["Ticker"] != ""].copy()
            return filtered.reset_index(drop=True)

    raise MissingDataError("Could not find the Holdings table in Investments.numbers.")


@st.cache_data(persist="disk", show_spinner=False)
def load_portfolio_holdings() -> pd.DataFrame:
    holdings = _load_holdings_table_from_numbers(PORTFOLIO_DOCUMENT_PATH)
    return holdings


def portfolio_history_tickers() -> list[str]:
    holdings = load_portfolio_holdings()
    if holdings.empty:
        return []
    tickers = holdings.loc[
        holdings["Type"].astype(str).str.strip().str.lower() != "cash",
        "Ticker",
    ]
    return [ticker for ticker in tickers.astype(str).str.strip().str.upper().tolist() if ticker]


def _coerce_float(value):
    if value in ("", None) or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_date(value) -> date | None:
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _normalize_yield(value):
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    if 0.15 <= numeric:
        return numeric / 100
    return numeric


def _compute_reinvested_asset_value(
    price_series: pd.Series,
    income_per_share_series: pd.Series,
    initial_quantity: float,
) -> pd.Series:
    shares = initial_quantity
    values: list[float] = []

    for timestamp, price in price_series.items():
        if pd.isna(price) or price in (None, 0):
            values.append(0.0)
            continue

        income_per_share = float(income_per_share_series.get(timestamp, 0.0) or 0.0)
        if income_per_share:
            shares += shares * income_per_share / float(price)
        values.append(shares * float(price))

    return pd.Series(values, index=price_series.index, dtype="float64")


def _compute_reinvested_cash_value(index: pd.DatetimeIndex, principal: float) -> pd.Series:
    values: list[float] = []
    current_value = principal
    daily_rate = CASH_YIELD / 365.0

    for _ in index:
        values.append(current_value)
        current_value *= 1.0 + daily_rate

    return pd.Series(values, index=index, dtype="float64")


@st.cache_data(ttl=PORTFOLIO_CACHE_TTL_SECONDS, show_spinner=False)
def build_holdings_analysis_table() -> pd.DataFrame:
    source = load_portfolio_holdings().copy()

    source["Quantity"] = source["Quantity"].map(_coerce_float)
    source["Bought at"] = source["Bought at"].map(_coerce_float)

    descriptions = []
    prices = []
    yields = []
    pes = []
    market_values = []
    gains = []
    incomes = []

    for _, row in source.iterrows():
        ticker = row["Ticker"]
        asset_type = str(row["Type"]).strip().lower()
        quantity = row["Quantity"] or 0.0
        bought_at = row["Bought at"] or 0.0

        if asset_type == "cash":
            description = "Cash"
            price = CASH_PRICE
            dividend_yield = CASH_YIELD
            trailing_pe = None
        else:
            try:
                snapshot, _, _ = get_ticker_snapshot(ticker)
                description = snapshot.short_name
                price = snapshot.regular_market_price
                dividend_yield = _normalize_yield(snapshot.dividend_yield)
                trailing_pe = snapshot.trailing_pe
            except Exception:
                description = ""
                price = None
                dividend_yield = None
                trailing_pe = None

        market_value = quantity * price if price is not None else None
        gain = ((price - bought_at) / bought_at) if price is not None and bought_at not in (None, 0) else None
        income = (
            dividend_yield * market_value
            if dividend_yield is not None and market_value is not None
            else None
        )

        descriptions.append(description)
        prices.append(price)
        yields.append(dividend_yield)
        pes.append(trailing_pe)
        market_values.append(market_value)
        gains.append(gain)
        incomes.append(income)

    source["Description"] = descriptions
    source["Price"] = prices
    source["Yield"] = yields
    source["P/E"] = pes
    source["Market Value"] = market_values
    source["Gain"] = gains
    source["Income"] = incomes

    ordered_columns = [
        "Ticker",
        "Type",
        "Loc",
        "Description",
        "Quantity",
        "Buy Date",
        "Bought at",
        "Price",
        "Market Value",
        "Gain",
        "Yield",
        "P/E",
        "Income",
    ]
    return source[ordered_columns]


@st.cache_data(ttl=PORTFOLIO_CACHE_TTL_SECONDS, show_spinner=False)
def build_holdings_portfolio_histories() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    holdings = load_portfolio_holdings().copy()
    if holdings.empty:
        empty_value = pd.DataFrame(columns=["Portfolio Value"])
        empty_income = pd.DataFrame(columns=["Income"])
        return {
            "actual": (empty_value.copy(), empty_income.copy()),
            "full_year": (empty_value.copy(), empty_income.copy()),
        }

    holdings["Quantity"] = holdings["Quantity"].map(_coerce_float)
    holdings["Buy Date"] = holdings["Buy Date"].map(_coerce_date)

    end_date = date.today()
    history_starts: list[date] = []
    non_cash_histories: dict[str, pd.DataFrame] = {}
    for _, row in holdings.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        asset_type = str(row.get("Type", "")).strip().lower()
        buy_date = row.get("Buy Date")

        if asset_type == "cash":
            if buy_date:
                history_starts.append(buy_date)
            continue
        if not ticker:
            continue
        try:
            history = get_full_price_history(ticker)
        except Exception:
            continue
        if history.empty:
            continue
        non_cash_histories[ticker] = history
        history_starts.append(history.index.min().date())
        if buy_date:
            history_starts.append(buy_date)

    if not history_starts:
        empty_value = pd.DataFrame(columns=["Portfolio Value"])
        empty_income = pd.DataFrame(columns=["Income"])
        return {
            "actual": (empty_value.copy(), empty_income.copy()),
            "full_year": (empty_value.copy(), empty_income.copy()),
        }

    start_date = min(history_starts)
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    actual_total_value = pd.Series(0.0, index=full_index, dtype="float64")
    actual_total_income = pd.Series(0.0, index=full_index, dtype="float64")
    actual_total_reinvested_value = pd.Series(0.0, index=full_index, dtype="float64")
    full_year_total_value = pd.Series(0.0, index=full_index, dtype="float64")
    full_year_total_income = pd.Series(0.0, index=full_index, dtype="float64")
    full_year_total_reinvested_value = pd.Series(0.0, index=full_index, dtype="float64")

    for _, row in holdings.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        asset_type = str(row.get("Type", "")).strip().lower()
        quantity = row.get("Quantity") or 0.0
        buy_date = row.get("Buy Date")

        if not ticker or quantity <= 0:
            continue

        actual_start = max(start_date, buy_date) if buy_date else start_date
        if actual_start > end_date:
            continue

        if asset_type == "cash":
            full_year_asset_value = pd.Series(quantity * CASH_PRICE, index=full_index, dtype="float64")
            full_year_asset_income = pd.Series(
                quantity * CASH_PRICE * CASH_YIELD / 365.0,
                index=full_index,
                dtype="float64",
            )
            full_year_asset_reinvested_value = _compute_reinvested_cash_value(
                full_index,
                quantity * CASH_PRICE,
            )
        else:
            history = non_cash_histories.get(ticker)
            if history is None or history.empty:
                continue

            close_series = history["Close"].dropna()
            if close_series.empty:
                continue

            close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
            dividends = history["Dividends"].fillna(0.0)
            dividends.index = pd.to_datetime(dividends.index).tz_localize(None)

            full_year_asset_value = close_series.reindex(full_index).ffill().bfill() * quantity
            full_year_asset_income = dividends.reindex(full_index, fill_value=0.0) * quantity
            full_year_asset_reinvested_value = _compute_reinvested_asset_value(
                close_series.reindex(full_index).ffill().bfill(),
                dividends.reindex(full_index, fill_value=0.0),
                quantity,
            )

        actual_value = full_year_asset_value.copy()
        actual_income = full_year_asset_income.copy()
        actual_reinvested_value = full_year_asset_reinvested_value.copy()
        actual_value.loc[actual_value.index < pd.Timestamp(actual_start)] = 0.0
        actual_income.loc[actual_income.index < pd.Timestamp(actual_start)] = 0.0
        actual_reinvested_value.loc[actual_reinvested_value.index < pd.Timestamp(actual_start)] = 0.0

        if actual_start > start_date:
            actual_index = pd.date_range(start=actual_start, end=end_date, freq="D")
            if asset_type == "cash":
                actual_reinvested_slice = _compute_reinvested_cash_value(
                    actual_index,
                    quantity * CASH_PRICE,
                )
            else:
                actual_reinvested_slice = _compute_reinvested_asset_value(
                    close_series.reindex(actual_index).ffill().bfill(),
                    dividends.reindex(actual_index, fill_value=0.0),
                    quantity,
                )
            actual_reinvested_value.loc[actual_index] = actual_reinvested_slice

        actual_total_value = actual_total_value.add(actual_value, fill_value=0.0)
        actual_total_income = actual_total_income.add(actual_income, fill_value=0.0)
        actual_total_reinvested_value = actual_total_reinvested_value.add(
            actual_reinvested_value,
            fill_value=0.0,
        )
        full_year_total_value = full_year_total_value.add(full_year_asset_value, fill_value=0.0)
        full_year_total_income = full_year_total_income.add(full_year_asset_income, fill_value=0.0)
        full_year_total_reinvested_value = full_year_total_reinvested_value.add(
            full_year_asset_reinvested_value,
            fill_value=0.0,
        )

    actual_monthly_income = actual_total_income.resample("ME").sum()
    actual_monthly_income = actual_monthly_income.loc[actual_monthly_income.index <= pd.Timestamp(end_date)]
    full_year_monthly_income = full_year_total_income.resample("ME").sum()
    full_year_monthly_income = full_year_monthly_income.loc[
        full_year_monthly_income.index <= pd.Timestamp(end_date)
    ]

    return {
        "actual": (
            pd.DataFrame(
                {
                    "Portfolio Value": actual_total_value,
                    "Reinvested Portfolio Value": actual_total_reinvested_value,
                }
            ),
            pd.DataFrame({"Income": actual_monthly_income}),
        ),
        "full_year": (
            pd.DataFrame(
                {
                    "Portfolio Value": full_year_total_value,
                    "Reinvested Portfolio Value": full_year_total_reinvested_value,
                }
            ),
            pd.DataFrame({"Income": full_year_monthly_income}),
        ),
    }


def compute_holdings_totals(holdings: pd.DataFrame) -> dict[str, float]:
    return {
        "market_value": float(pd.to_numeric(holdings["Market Value"], errors="coerce").fillna(0.0).sum()),
        "income": float(pd.to_numeric(holdings["Income"], errors="coerce").fillna(0.0).sum()),
    }


def _format_currency(value) -> str:
    if value in ("", None) or pd.isna(value):
        return ""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value) -> str:
    if value in ("", None) or pd.isna(value):
        return ""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_integer(value) -> str:
    if value in ("", None) or pd.isna(value):
        return ""
    try:
        return f"{round(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def format_portfolio_holdings_for_display(holdings: pd.DataFrame) -> pd.DataFrame:
    display = holdings.copy()

    for column in ["Quantity", "P/E"]:
        if column in display.columns:
            display[column] = display[column].map(_format_integer)

    for column in ["Bought at", "Price", "Price  ", "Market Value", "Income"]:
        if column in display.columns:
            display[column] = display[column].map(_format_currency)

    for column in ["Gain", "Yield"]:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)

    return display


def refresh_holdings_analysis_data() -> None:
    load_portfolio_holdings.clear()
    build_holdings_analysis_table.clear()
    build_holdings_portfolio_histories.clear()
