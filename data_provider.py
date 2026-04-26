from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

from errors import InvalidTickerError, ProviderError
from models import TickerSnapshot


CACHE_DIR = Path(".cache/price_history")
CACHE_EXTENSION = ".csv"
CACHE_TTL_SECONDS = 86400


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        if value == "":
            continue
        return value
    return None


def _normalize_financial_statement(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    normalized.columns = pd.to_datetime(normalized.columns).tz_localize(None)
    return normalized


def _normalize_history_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Close", "Adj Close", "Dividends"])

    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None)

    for column in ["Close", "Adj Close", "Dividends"]:
        if column not in normalized.columns:
            normalized[column] = 0.0 if column == "Dividends" else pd.NA

    return normalized.sort_index()


def _cache_path_for(symbol: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol.lower()}{CACHE_EXTENSION}"


def _read_cached_history(symbol: str) -> pd.DataFrame:
    path = _cache_path_for(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["Close", "Adj Close", "Dividends"])

    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame(columns=["Close", "Adj Close", "Dividends"])

    return _normalize_history_frame(cached)


def _write_cached_history(symbol: str, history: pd.DataFrame) -> None:
    path = _cache_path_for(symbol)
    history_to_save = history.copy()
    history_to_save.index = pd.to_datetime(history_to_save.index).tz_localize(None)
    history_to_save.to_csv(path)


def _download_history_segment(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    fetch_end = end_date + timedelta(days=1)
    return _normalize_history_frame(
        yf.Ticker(symbol).history(start=start_date, end=fetch_end, auto_adjust=False)
    )


def _slice_history(history: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if history.empty:
        return history
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return history.loc[(history.index >= start_ts) & (history.index <= end_ts)].copy()


def _refresh_cached_history(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    cached = _read_cached_history(symbol)

    segments: list[pd.DataFrame] = []
    if not cached.empty:
        cached = _slice_history(cached, min(start_date, cached.index.min().date()), end_date)

    if cached.empty:
        refreshed = _download_history_segment(symbol, start_date, end_date)
        if not refreshed.empty:
            _write_cached_history(symbol, refreshed)
        return _slice_history(refreshed, start_date, end_date)

    cached_start = cached.index.min().date()
    cached_end = cached.index.max().date()

    if start_date < cached_start:
        segments.append(_download_history_segment(symbol, start_date, cached_start - timedelta(days=1)))

    if cached_end < end_date:
        update_start = max(start_date, cached_end - timedelta(days=7))
        segments.append(_download_history_segment(symbol, update_start, end_date))

    if segments:
        combined = pd.concat([cached] + [segment for segment in segments if not segment.empty], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        _write_cached_history(symbol, combined)
        cached = combined

    return _slice_history(cached, start_date, end_date)


def _extract_fast_info(stock: yf.Ticker) -> dict[str, Any]:
    try:
        fast_info = stock.fast_info
        if hasattr(fast_info, "items"):
            return dict(fast_info.items())
        return dict(fast_info)
    except Exception:
        return {}


def _build_snapshot(ticker: str, info: dict[str, Any], fast_info: dict[str, Any]) -> TickerSnapshot:
    regular_market_price = _safe_float(
        _first_non_null(
            info.get("regularMarketPrice"),
            info.get("currentPrice"),
            info.get("navPrice"),
            fast_info.get("lastPrice"),
            fast_info.get("regularMarketPreviousClose"),
            info.get("previousClose"),
        )
    )
    shares_outstanding = _safe_float(
        _first_non_null(
            info.get("sharesOutstanding"),
            fast_info.get("shares"),
        )
    )
    if shares_outstanding is None and regular_market_price:
        market_cap = _safe_float(info.get("marketCap"))
        if market_cap:
            shares_outstanding = market_cap / regular_market_price

    return TickerSnapshot(
        ticker=ticker,
        short_name=str(
            _first_non_null(
                info.get("shortName"),
                info.get("longName"),
                info.get("displayName"),
                ticker,
            )
        ),
        asset_type=str(_first_non_null(info.get("quoteType"), "UNKNOWN")),
        regular_market_price=regular_market_price,
        dividend_yield=_safe_float(info.get("dividendYield")),
        trailing_pe=_safe_float(info.get("trailingPE")),
        shares_outstanding=shares_outstanding,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_ticker_snapshot(ticker: str) -> tuple[TickerSnapshot, pd.DataFrame]:
    symbol = ticker.strip().upper()
    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
        fast_info = _extract_fast_info(stock)
        quarterly_income_stmt = _normalize_financial_statement(stock.quarterly_income_stmt)
        sample_history = _normalize_history_frame(stock.history(period="1mo", auto_adjust=False))
    except Exception as exc:
        raise ProviderError(f"Could not retrieve data for {symbol}.", str(exc)) from exc

    snapshot = _build_snapshot(symbol, info, fast_info)

    if snapshot.regular_market_price is None and not sample_history.empty:
        latest_close = sample_history["Close"].dropna()
        if not latest_close.empty:
            snapshot.regular_market_price = _safe_float(latest_close.iloc[-1])

    if (
        snapshot.regular_market_price is None
        and quarterly_income_stmt.empty
        and sample_history.empty
        and snapshot.short_name == symbol
    ):
        raise InvalidTickerError(f"Ticker '{symbol}' was not found or returned no usable data.")

    return snapshot, quarterly_income_stmt


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_price_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    symbol = ticker.strip().upper()
    try:
        return _refresh_cached_history(symbol, start_date, end_date)
    except Exception as exc:
        raise ProviderError(
            f"Could not retrieve price history for {symbol}.",
            str(exc),
        ) from exc


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_daily_close_history(ticker: str, start_date: date, end_date: date) -> pd.Series:
    symbol = ticker.strip().upper()
    try:
        history = _refresh_cached_history(symbol, start_date, end_date)
    except Exception as exc:
        raise ProviderError(
            f"Could not retrieve daily prices for {symbol}.",
            str(exc),
        ) from exc

    close_series = history["Close"].dropna()
    close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
    return close_series
