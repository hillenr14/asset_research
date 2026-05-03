from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from dataclasses import asdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

from errors import InvalidTickerError, ProviderError
from models import TickerSnapshot


CACHE_DIR = Path(".cache/price_history")
CACHE_EXTENSION = ".csv"
SNAPSHOT_CACHE_DIR = Path(".cache/ticker_snapshots")
SNAPSHOT_CACHE_EXTENSION = ".json"
INCOME_STATEMENT_CACHE_EXTENSION = ".statement.csv"
CASHFLOW_STATEMENT_CACHE_EXTENSION = ".cashflow.csv"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16
NYSE_CALENDAR = xcals.get_calendar("XNYS")
FULL_HISTORY_MEMORY_CACHE: dict[str, pd.DataFrame] = {}


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


def _snapshot_cache_paths_for(symbol: str) -> tuple[Path, Path, Path]:
    SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_name = symbol.lower()
    return (
        SNAPSHOT_CACHE_DIR / f"{base_name}{SNAPSHOT_CACHE_EXTENSION}",
        SNAPSHOT_CACHE_DIR / f"{base_name}{INCOME_STATEMENT_CACHE_EXTENSION}",
        SNAPSHOT_CACHE_DIR / f"{base_name}{CASHFLOW_STATEMENT_CACHE_EXTENSION}",
    )


def _read_cached_history(symbol: str) -> pd.DataFrame:
    path = _cache_path_for(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["Close", "Adj Close", "Dividends"])

    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame(columns=["Close", "Adj Close", "Dividends"])

    return _normalize_history_frame(cached)


def _latest_expected_history_date() -> date:
    now_market = datetime.now(MARKET_TIMEZONE)
    market_close_today = now_market.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    today = pd.Timestamp(now_market.date())

    if NYSE_CALENDAR.is_session(today) and now_market >= market_close_today:
        return today.date()

    previous_session = NYSE_CALENDAR.date_to_session(today, direction="previous")
    return previous_session.date()


def _cached_history_is_fresh(cached_end: date) -> bool:
    return cached_end >= _latest_expected_history_date()


def _snapshot_path_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    file_date = datetime.fromtimestamp(path.stat().st_mtime, tz=MARKET_TIMEZONE).date()
    return file_date >= _latest_expected_history_date()


def _write_cached_history(symbol: str, history: pd.DataFrame) -> None:
    path = _cache_path_for(symbol)
    history_to_save = history.copy()
    history_to_save.index = pd.to_datetime(history_to_save.index).tz_localize(None)
    history_to_save.to_csv(path)


def _read_cached_snapshot(symbol: str) -> tuple[TickerSnapshot, pd.DataFrame, pd.DataFrame] | None:
    snapshot_path, income_statement_path, cashflow_statement_path = _snapshot_cache_paths_for(symbol)
    if not (
        _snapshot_path_is_fresh(snapshot_path)
        and _snapshot_path_is_fresh(income_statement_path)
        and _snapshot_path_is_fresh(cashflow_statement_path)
    ):
        return None

    try:
        snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot = TickerSnapshot(**snapshot_payload)
        if income_statement_path.stat().st_size <= 1:
            income_statement = pd.DataFrame()
        else:
            income_statement = pd.read_csv(income_statement_path, index_col=0)
            income_statement.columns = pd.to_datetime(income_statement.columns).tz_localize(None)
            income_statement = _normalize_financial_statement(income_statement)
        if cashflow_statement_path.stat().st_size <= 1:
            cashflow_statement = pd.DataFrame()
        else:
            cashflow_statement = pd.read_csv(cashflow_statement_path, index_col=0)
            cashflow_statement.columns = pd.to_datetime(cashflow_statement.columns).tz_localize(None)
            cashflow_statement = _normalize_financial_statement(cashflow_statement)
    except Exception:
        return None

    return snapshot, income_statement, cashflow_statement


def _write_cached_snapshot(
    symbol: str,
    snapshot: TickerSnapshot,
    quarterly_income_stmt: pd.DataFrame,
    quarterly_cashflow_stmt: pd.DataFrame,
) -> None:
    snapshot_path, income_statement_path, cashflow_statement_path = _snapshot_cache_paths_for(symbol)
    snapshot_path.write_text(json.dumps(asdict(snapshot)), encoding="utf-8")
    _normalize_financial_statement(quarterly_income_stmt).to_csv(income_statement_path)
    _normalize_financial_statement(quarterly_cashflow_stmt).to_csv(cashflow_statement_path)


def _download_full_history(symbol: str) -> pd.DataFrame:
    return _normalize_history_frame(
        yf.Ticker(symbol).history(period="max", auto_adjust=False)
    )


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


def _load_full_cached_history(symbol: str, end_date: date) -> pd.DataFrame:
    in_memory = FULL_HISTORY_MEMORY_CACHE.get(symbol)
    if in_memory is not None and not in_memory.empty:
        cached_end = in_memory.index.max().date()
        if _cached_history_is_fresh(cached_end):
            return in_memory

    cached = _read_cached_history(symbol)
    if not cached.empty:
        cached = cached.sort_index()

    if cached.empty:
        refreshed = _download_full_history(symbol)
        if not refreshed.empty:
            _write_cached_history(symbol, refreshed)
            FULL_HISTORY_MEMORY_CACHE[symbol] = refreshed
        return refreshed

    cached_end = cached.index.max().date()
    cache_is_fresh = _cached_history_is_fresh(cached_end)

    non_empty_segments: list[pd.DataFrame] = []
    if not cache_is_fresh and cached_end < end_date:
        update_start = cached_end - timedelta(days=7)
        segment = _download_history_segment(symbol, update_start, end_date)
        if not segment.empty:
            non_empty_segments.append(segment)

    if non_empty_segments:
        combined = pd.concat([cached] + non_empty_segments, axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        _write_cached_history(symbol, combined)
        cached = combined

    FULL_HISTORY_MEMORY_CACHE[symbol] = cached
    return cached


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


def get_ticker_snapshot(ticker: str) -> tuple[TickerSnapshot, pd.DataFrame, pd.DataFrame]:
    symbol = ticker.strip().upper()
    cached = _read_cached_snapshot(symbol)
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
        fast_info = _extract_fast_info(stock)
        quarterly_income_stmt = _normalize_financial_statement(stock.quarterly_income_stmt)
        quarterly_cashflow_stmt = _normalize_financial_statement(stock.quarterly_cashflow)
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
        and quarterly_cashflow_stmt.empty
        and sample_history.empty
        and snapshot.short_name == symbol
    ):
        raise InvalidTickerError(f"Ticker '{symbol}' was not found or returned no usable data.")

    _write_cached_snapshot(symbol, snapshot, quarterly_income_stmt, quarterly_cashflow_stmt)
    return snapshot, quarterly_income_stmt, quarterly_cashflow_stmt


def warm_price_history_cache(tickers: list[str]) -> None:
    end_date = date.today()
    normalized_tickers: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized_tickers.append(symbol)

    for symbol in normalized_tickers:
        _load_full_cached_history(symbol, end_date)


def clear_in_memory_price_history_cache() -> None:
    FULL_HISTORY_MEMORY_CACHE.clear()


def get_full_price_history(ticker: str) -> pd.DataFrame:
    symbol = ticker.strip().upper()
    try:
        return _load_full_cached_history(symbol, date.today()).copy()
    except Exception as exc:
        raise ProviderError(
            f"Could not retrieve full price history for {symbol}.",
            str(exc),
        ) from exc


def get_price_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    try:
        history = get_full_price_history(ticker)
        return _slice_history(history, start_date, end_date)
    except Exception as exc:
        raise ProviderError(
            f"Could not retrieve price history for {ticker.strip().upper()}.",
            str(exc),
        ) from exc


def get_daily_close_history(ticker: str, start_date: date, end_date: date) -> pd.Series:
    try:
        history = get_full_price_history(ticker)
        history = _slice_history(history, start_date, end_date)
    except Exception as exc:
        raise ProviderError(
            f"Could not retrieve daily prices for {ticker.strip().upper()}.",
            str(exc),
        ) from exc

    close_series = history["Close"].dropna()
    close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
    return close_series
