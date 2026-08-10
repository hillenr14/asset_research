from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

from errors import InvalidTickerError, ProviderError
from models import TickerSnapshot
from ticker_symbols import normalize_ticker_symbol


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
SNAPSHOT_MEMORY_CACHE: dict[str, tuple[TickerSnapshot, pd.DataFrame, pd.DataFrame]] = {}
HISTORY_REFRESH_ATTEMPTS: dict[str, tuple[date, datetime]] = {}
HISTORY_REFRESH_ATTEMPT_LOCK = Lock()
SNAPSHOT_REFRESH_ATTEMPTS: dict[tuple[str, str], datetime] = {}
SNAPSHOT_REFRESH_ATTEMPT_LOCK = Lock()
HISTORY_REFRESH_BACKOFF = timedelta(minutes=20)
SNAPSHOT_REFRESH_BACKOFF = timedelta(minutes=20)
STATEMENT_CACHE_TTL = timedelta(days=7)
MAX_HISTORY_REFRESH_WORKERS = 8


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
    symbol = normalize_ticker_symbol(symbol)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol.lower()}{CACHE_EXTENSION}"
    if path.resolve().parent != CACHE_DIR.resolve():
        raise InvalidTickerError(f"Ticker '{symbol}' cannot be used as a cache key.")
    return path


def _snapshot_cache_paths_for(symbol: str) -> tuple[Path, Path, Path]:
    symbol = normalize_ticker_symbol(symbol)
    SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_name = symbol.lower()
    paths = (
        SNAPSHOT_CACHE_DIR / f"{base_name}{SNAPSHOT_CACHE_EXTENSION}",
        SNAPSHOT_CACHE_DIR / f"{base_name}{INCOME_STATEMENT_CACHE_EXTENSION}",
        SNAPSHOT_CACHE_DIR / f"{base_name}{CASHFLOW_STATEMENT_CACHE_EXTENSION}",
    )
    cache_root = SNAPSHOT_CACHE_DIR.resolve()
    if any(path.resolve().parent != cache_root for path in paths):
        raise InvalidTickerError(f"Ticker '{symbol}' cannot be used as a cache key.")
    return paths


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


def latest_expected_history_date() -> date:
    """Return the most recent NYSE session expected to have closing data."""
    return _latest_expected_history_date()


def _cached_history_is_fresh(cached_end: date) -> bool:
    return cached_end >= _latest_expected_history_date()


def _market_close_for_session(session_date: date) -> datetime:
    return datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        MARKET_CLOSE_HOUR,
        0,
        0,
        tzinfo=MARKET_TIMEZONE,
    )


def _snapshot_path_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    snapshot_timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=MARKET_TIMEZONE)
    required_close_timestamp = _market_close_for_session(_latest_expected_history_date())
    return snapshot_timestamp >= required_close_timestamp


def _statement_path_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=MARKET_TIMEZONE)
    return datetime.now(MARKET_TIMEZONE) - modified <= STATEMENT_CACHE_TTL


def _claim_history_refresh(symbol: str, expected_session: date) -> bool:
    """Atomically reserve a refresh unless this session is in its cooldown."""
    with HISTORY_REFRESH_ATTEMPT_LOCK:
        attempted = HISTORY_REFRESH_ATTEMPTS.get(symbol)
        now = datetime.now(MARKET_TIMEZONE)
        if (
            attempted is not None
            and attempted[0] == expected_session
            and now - attempted[1] < HISTORY_REFRESH_BACKOFF
        ):
            return False
        HISTORY_REFRESH_ATTEMPTS[symbol] = (expected_session, now)
        return True


def _clear_history_refresh_attempt(symbol: str) -> None:
    with HISTORY_REFRESH_ATTEMPT_LOCK:
        HISTORY_REFRESH_ATTEMPTS.pop(symbol, None)


def _claim_snapshot_refresh(symbol: str, component: str) -> bool:
    """Reserve one snapshot-component request during the cooldown window."""
    key = (symbol, component)
    with SNAPSHOT_REFRESH_ATTEMPT_LOCK:
        attempted_at = SNAPSHOT_REFRESH_ATTEMPTS.get(key)
        now = datetime.now(MARKET_TIMEZONE)
        if (
            attempted_at is not None
            and now - attempted_at < SNAPSHOT_REFRESH_BACKOFF
        ):
            return False
        SNAPSHOT_REFRESH_ATTEMPTS[key] = now
        return True


def _clear_snapshot_refresh_attempt(symbol: str, component: str) -> None:
    with SNAPSHOT_REFRESH_ATTEMPT_LOCK:
        SNAPSHOT_REFRESH_ATTEMPTS.pop((symbol, component), None)


def _write_cached_history(symbol: str, history: pd.DataFrame) -> None:
    path = _cache_path_for(symbol)
    history_to_save = history.copy()
    history_to_save.index = pd.to_datetime(history_to_save.index).tz_localize(None)
    history_to_save.to_csv(path)


def _read_cached_snapshot(symbol: str) -> tuple[TickerSnapshot, pd.DataFrame, pd.DataFrame] | None:
    snapshot_path, income_statement_path, cashflow_statement_path = _snapshot_cache_paths_for(symbol)
    if not (
        _snapshot_path_is_fresh(snapshot_path)
        and _statement_path_is_fresh(income_statement_path)
        and _statement_path_is_fresh(cashflow_statement_path)
    ):
        return None

    in_memory = SNAPSHOT_MEMORY_CACHE.get(symbol)
    if in_memory is not None:
        return in_memory

    cached = _read_cached_snapshot_components(symbol)
    if cached[0] is None or cached[1] is None or cached[2] is None:
        return None
    result = cached[0], cached[1], cached[2]
    SNAPSHOT_MEMORY_CACHE[symbol] = result
    return result


def _read_cached_snapshot_components(
    symbol: str,
) -> tuple[TickerSnapshot | None, pd.DataFrame | None, pd.DataFrame | None]:
    snapshot_path, income_statement_path, cashflow_statement_path = _snapshot_cache_paths_for(symbol)

    try:
        snapshot = TickerSnapshot(**json.loads(snapshot_path.read_text(encoding="utf-8")))
    except Exception:
        snapshot = None

    def read_statement(path: Path) -> pd.DataFrame | None:
        try:
            if path.stat().st_size <= 1:
                return pd.DataFrame()
            statement = pd.read_csv(path, index_col=0)
            statement.columns = pd.to_datetime(statement.columns).tz_localize(None)
            return _normalize_financial_statement(statement)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            return None

    return snapshot, read_statement(income_statement_path), read_statement(cashflow_statement_path)


def get_cached_ticker_name(ticker: str) -> str | None:
    """Return a cached display name without refreshing provider data."""
    symbol = normalize_ticker_symbol(ticker)
    snapshot, _, _ = _read_cached_snapshot_components(symbol)
    if snapshot is None or not snapshot.short_name or snapshot.short_name == symbol:
        return None
    return snapshot.short_name


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
    SNAPSHOT_MEMORY_CACHE[symbol] = (
        snapshot,
        _normalize_financial_statement(quarterly_income_stmt),
        _normalize_financial_statement(quarterly_cashflow_stmt),
    )


def _write_snapshot_component(symbol: str, snapshot: TickerSnapshot) -> None:
    snapshot_path, _, _ = _snapshot_cache_paths_for(symbol)
    snapshot_path.write_text(json.dumps(asdict(snapshot)), encoding="utf-8")


def _write_statement_component(symbol: str, statement: pd.DataFrame, *, cashflow: bool) -> None:
    _, income_statement_path, cashflow_statement_path = _snapshot_cache_paths_for(symbol)
    path = cashflow_statement_path if cashflow else income_statement_path
    _normalize_financial_statement(statement).to_csv(path)


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
    symbol = normalize_ticker_symbol(symbol)
    in_memory = FULL_HISTORY_MEMORY_CACHE.get(symbol)
    if in_memory is not None and not in_memory.empty:
        cached_end = in_memory.index.max().date()
        if _cached_history_is_fresh(cached_end):
            return in_memory

    cached = _read_cached_history(symbol)
    if not cached.empty:
        cached = cached.sort_index()

    if cached.empty:
        expected_session = _latest_expected_history_date()
        if not _claim_history_refresh(symbol, expected_session):
            return cached
        try:
            refreshed = _download_full_history(symbol)
        except Exception:
            # Keep the claim in place so an unavailable provider is not queried
            # again by every downstream consumer in the same app session.
            raise
        if not refreshed.empty:
            _write_cached_history(symbol, refreshed)
            FULL_HISTORY_MEMORY_CACHE[symbol] = refreshed
            if refreshed.index.max().date() >= expected_session:
                _clear_history_refresh_attempt(symbol)
        return refreshed

    cached_end = cached.index.max().date()
    expected_session = _latest_expected_history_date()
    cache_is_fresh = cached_end >= expected_session

    non_empty_segments: list[pd.DataFrame] = []
    if (
        not cache_is_fresh
        and cached_end < end_date
        and _claim_history_refresh(symbol, expected_session)
    ):
        update_start = cached_end - timedelta(days=7)
        try:
            segment = _download_history_segment(symbol, update_start, end_date)
        except Exception:
            segment = pd.DataFrame()
        if not segment.empty:
            non_empty_segments.append(segment)

    if non_empty_segments:
        combined = pd.concat([cached] + non_empty_segments, axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        _write_cached_history(symbol, combined)
        cached = combined

    if not cached.empty and cached.index.max().date() >= expected_session:
        _clear_history_refresh_attempt(symbol)

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
    symbol = normalize_ticker_symbol(ticker)
    cached = _read_cached_snapshot(symbol)
    if cached is not None:
        return cached

    snapshot_path, income_statement_path, cashflow_statement_path = _snapshot_cache_paths_for(symbol)
    stale_snapshot, stale_income, stale_cashflow = _read_cached_snapshot_components(symbol)
    in_memory = SNAPSHOT_MEMORY_CACHE.get(symbol)
    if in_memory is not None:
        stale_snapshot, stale_income, stale_cashflow = in_memory
    quote_is_fresh = _snapshot_path_is_fresh(snapshot_path) and stale_snapshot is not None
    income_is_fresh = _statement_path_is_fresh(income_statement_path) and stale_income is not None
    cashflow_is_fresh = _statement_path_is_fresh(cashflow_statement_path) and stale_cashflow is not None

    refresh_quote = not quote_is_fresh and _claim_snapshot_refresh(symbol, "quote")
    refresh_income = not income_is_fresh and _claim_snapshot_refresh(symbol, "income")
    refresh_cashflow = not cashflow_is_fresh and _claim_snapshot_refresh(symbol, "cashflow")
    stock: yf.Ticker | None = None
    if refresh_quote or refresh_income or refresh_cashflow:
        try:
            stock = yf.Ticker(symbol)
        except Exception as exc:
            if stale_snapshot is None:
                raise ProviderError(f"Could not retrieve data for {symbol}.", str(exc)) from exc

    snapshot = stale_snapshot
    snapshot_was_refreshed = False
    if refresh_quote and stock is not None:
        try:
            info = stock.info or {}
            fast_info = _extract_fast_info(stock)
            refreshed_snapshot = _build_snapshot(symbol, info, fast_info)
            if refreshed_snapshot.regular_market_price is None:
                sample_history = _normalize_history_frame(
                    stock.history(period="1mo", auto_adjust=False)
                )
                if not sample_history.empty:
                    latest_close = sample_history["Close"].dropna()
                    if not latest_close.empty:
                        refreshed_snapshot.regular_market_price = _safe_float(latest_close.iloc[-1])
            snapshot = refreshed_snapshot
            snapshot_was_refreshed = True
        except Exception:
            if snapshot is None:
                raise ProviderError(f"Could not retrieve quote data for {symbol}.")

    quarterly_income_stmt = stale_income
    if refresh_income and stock is not None:
        try:
            quarterly_income_stmt = _normalize_financial_statement(stock.quarterly_income_stmt)
            _write_statement_component(symbol, quarterly_income_stmt, cashflow=False)
            _clear_snapshot_refresh_attempt(symbol, "income")
        except Exception:
            if quarterly_income_stmt is None:
                quarterly_income_stmt = pd.DataFrame()

    quarterly_cashflow_stmt = stale_cashflow
    if refresh_cashflow and stock is not None:
        try:
            quarterly_cashflow_stmt = _normalize_financial_statement(stock.quarterly_cashflow)
            _write_statement_component(symbol, quarterly_cashflow_stmt, cashflow=True)
            _clear_snapshot_refresh_attempt(symbol, "cashflow")
        except Exception:
            if quarterly_cashflow_stmt is None:
                quarterly_cashflow_stmt = pd.DataFrame()

    if snapshot is None:
        raise ProviderError(f"Could not retrieve data for {symbol}.")
    quarterly_income_stmt = quarterly_income_stmt if quarterly_income_stmt is not None else pd.DataFrame()
    quarterly_cashflow_stmt = quarterly_cashflow_stmt if quarterly_cashflow_stmt is not None else pd.DataFrame()

    if (
        snapshot.regular_market_price is None
        and quarterly_income_stmt.empty
        and quarterly_cashflow_stmt.empty
        and snapshot.short_name == symbol
    ):
        raise InvalidTickerError(f"Ticker '{symbol}' was not found or returned no usable data.")

    if snapshot_was_refreshed:
        _write_snapshot_component(symbol, snapshot)
        _clear_snapshot_refresh_attempt(symbol, "quote")
    result = snapshot, quarterly_income_stmt, quarterly_cashflow_stmt
    SNAPSHOT_MEMORY_CACHE[symbol] = result
    return result


def warm_price_history_cache(tickers: list[str]) -> None:
    end_date = date.today()
    normalized_tickers: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if not ticker.strip():
            continue
        symbol = normalize_ticker_symbol(ticker)
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized_tickers.append(symbol)

    if not normalized_tickers:
        return
    worker_count = min(MAX_HISTORY_REFRESH_WORKERS, len(normalized_tickers))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="history-refresh") as executor:
        futures = {
            executor.submit(_load_full_cached_history, symbol, end_date): symbol
            for symbol in normalized_tickers
        }
        for future in as_completed(futures):
            # Preserve the prior best-effort warm-up behavior. Individual reads
            # still surface provider failures when the ticker is actually used.
            try:
                future.result()
            except Exception:
                continue


def clear_in_memory_price_history_cache() -> None:
    FULL_HISTORY_MEMORY_CACHE.clear()


def get_full_price_history(ticker: str) -> pd.DataFrame:
    symbol = normalize_ticker_symbol(ticker)
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
    except InvalidTickerError:
        raise
    except Exception as exc:
        raise ProviderError(
            f"Could not retrieve price history for {ticker.strip().upper()}.",
            str(exc),
        ) from exc


def get_daily_close_history(ticker: str, start_date: date, end_date: date) -> pd.Series:
    try:
        history = get_full_price_history(ticker)
        history = _slice_history(history, start_date, end_date)
    except InvalidTickerError:
        raise
    except Exception as exc:
        raise ProviderError(
            f"Could not retrieve daily prices for {ticker.strip().upper()}.",
            str(exc),
        ) from exc

    close_series = history["Close"].dropna()
    close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
    return close_series
