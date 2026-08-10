from __future__ import annotations

import os
from concurrent.futures import Future
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import data_provider
from errors import InvalidTickerError
from models import TickerSnapshot


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(data_provider, "CACHE_DIR", tmp_path / "history")
    monkeypatch.setattr(data_provider, "SNAPSHOT_CACHE_DIR", tmp_path / "snapshots")
    data_provider.FULL_HISTORY_MEMORY_CACHE.clear()
    data_provider.SNAPSHOT_MEMORY_CACHE.clear()
    data_provider.HISTORY_REFRESH_ATTEMPTS.clear()
    data_provider.SNAPSHOT_REFRESH_ATTEMPTS.clear()
    yield
    data_provider.FULL_HISTORY_MEMORY_CACHE.clear()
    data_provider.SNAPSHOT_MEMORY_CACHE.clear()
    data_provider.HISTORY_REFRESH_ATTEMPTS.clear()
    data_provider.SNAPSHOT_REFRESH_ATTEMPTS.clear()


def _history(ending: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0], "Adj Close": [99.0], "Dividends": [0.0]},
        index=pd.to_datetime([ending]),
    )


def _snapshot() -> TickerSnapshot:
    return TickerSnapshot(
        ticker="AAPL",
        short_name="Apple",
        asset_type="EQUITY",
        regular_market_price=200.0,
        dividend_yield=0.005,
        trailing_pe=30.0,
        shares_outstanding=15_000_000_000.0,
    )


def _statement(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [value]},
        index=["Total Revenue"],
    )


def test_empty_post_close_refresh_is_backed_off_for_expected_session(
    isolated_caches, monkeypatch
) -> None:
    expected = {"date": date(2026, 8, 7)}
    monkeypatch.setattr(
        data_provider, "_latest_expected_history_date", lambda: expected["date"]
    )
    data_provider._write_cached_history("AAPL", _history("2026-08-06"))
    calls: list[tuple[str, date, date]] = []

    def empty_download(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        calls.append((symbol, start_date, end_date))
        return pd.DataFrame()

    monkeypatch.setattr(data_provider, "_download_history_segment", empty_download)

    first = data_provider._load_full_cached_history("AAPL", expected["date"])
    second = data_provider._load_full_cached_history("AAPL", expected["date"])

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)

    expected["date"] = date(2026, 8, 10)
    data_provider._load_full_cached_history("AAPL", expected["date"])
    assert len(calls) == 2


def test_empty_full_download_is_backed_off_for_expected_session(
    isolated_caches, monkeypatch
) -> None:
    expected = date(2026, 8, 7)
    monkeypatch.setattr(data_provider, "_latest_expected_history_date", lambda: expected)
    calls: list[str] = []

    def empty_download(symbol: str) -> pd.DataFrame:
        calls.append(symbol)
        return pd.DataFrame()

    monkeypatch.setattr(data_provider, "_download_full_history", empty_download)

    first = data_provider._load_full_cached_history("NEW", expected)
    second = data_provider._load_full_cached_history("NEW", expected)

    assert first.empty
    assert second.empty
    assert calls == ["NEW"]


def test_warm_history_uses_bounded_workers_and_deduplicates(
    isolated_caches, monkeypatch
) -> None:
    submitted: list[str] = []
    worker_counts: list[int] = []

    class ImmediateExecutor:
        def __init__(self, max_workers: int, **_kwargs):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, symbol, end_date):
            submitted.append(symbol)
            future = Future()
            try:
                future.set_result(function(symbol, end_date))
            except Exception as exc:  # pragma: no cover - mirrors Executor behavior
                future.set_exception(exc)
            return future

    monkeypatch.setattr(data_provider, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(
        data_provider,
        "_load_full_cached_history",
        lambda symbol, _end_date: _history("2026-08-07"),
    )

    symbols = [f"T{number}" for number in range(10)] + ["T0", "t1"]
    data_provider.warm_price_history_cache(symbols)

    assert worker_counts == [data_provider.MAX_HISTORY_REFRESH_WORKERS]
    assert submitted == [f"T{number}" for number in range(10)]


def test_snapshot_refreshes_weekly_statements_without_refetching_fresh_quote(
    isolated_caches, monkeypatch
) -> None:
    data_provider._write_cached_snapshot(
        "AAPL", _snapshot(), _statement(1.0), _statement(2.0)
    )
    _, income_path, cashflow_path = data_provider._snapshot_cache_paths_for("AAPL")
    old_timestamp = (datetime.now() - timedelta(days=8)).timestamp()
    os.utime(income_path, (old_timestamp, old_timestamp))
    os.utime(cashflow_path, (old_timestamp, old_timestamp))

    class StatementsOnlyStock:
        @property
        def info(self):
            raise AssertionError("fresh quote should be reused")

        @property
        def quarterly_income_stmt(self):
            return _statement(11.0)

        @property
        def quarterly_cashflow(self):
            return _statement(22.0)

    monkeypatch.setattr(data_provider.yf, "Ticker", lambda _symbol: StatementsOnlyStock())

    snapshot, income, cashflow = data_provider.get_ticker_snapshot("aapl")

    assert snapshot.regular_market_price == 200.0
    assert income.iloc[0, 0] == 11.0
    assert cashflow.iloc[0, 0] == 22.0


def test_cached_ticker_name_does_not_refresh_provider_data(
    isolated_caches, monkeypatch
) -> None:
    data_provider._write_cached_snapshot(
        "AAPL", _snapshot(), _statement(1.0), _statement(2.0)
    )
    monkeypatch.setattr(
        data_provider.yf,
        "Ticker",
        lambda _symbol: pytest.fail("cached name lookup must not contact the provider"),
    )

    assert data_provider.get_cached_ticker_name("aapl") == "Apple"
    assert data_provider.get_cached_ticker_name("MSFT") is None


def test_snapshot_uses_stale_components_when_provider_temporarily_fails(
    isolated_caches, monkeypatch
) -> None:
    data_provider._write_cached_snapshot(
        "AAPL", _snapshot(), _statement(1.0), _statement(2.0)
    )
    for path in data_provider._snapshot_cache_paths_for("AAPL"):
        old_timestamp = (datetime.now() - timedelta(days=8)).timestamp()
        os.utime(path, (old_timestamp, old_timestamp))

    class FailingStock:
        @property
        def info(self):
            raise RuntimeError("provider unavailable")

        @property
        def quarterly_income_stmt(self):
            raise RuntimeError("provider unavailable")

        @property
        def quarterly_cashflow(self):
            raise RuntimeError("provider unavailable")

    provider_calls: list[str] = []

    def failing_ticker(symbol: str) -> FailingStock:
        provider_calls.append(symbol)
        return FailingStock()

    monkeypatch.setattr(data_provider.yf, "Ticker", failing_ticker)

    snapshot, income, cashflow = data_provider.get_ticker_snapshot("AAPL")
    repeated_snapshot, repeated_income, repeated_cashflow = data_provider.get_ticker_snapshot("AAPL")

    assert snapshot.short_name == "Apple"
    assert income.iloc[0, 0] == 1.0
    assert cashflow.iloc[0, 0] == 2.0
    assert repeated_snapshot.short_name == "Apple"
    assert repeated_income.iloc[0, 0] == 1.0
    assert repeated_cashflow.iloc[0, 0] == 2.0
    assert provider_calls == ["AAPL"]


def test_empty_statement_cache_is_reusable_after_process_memory_is_cleared(
    isolated_caches, monkeypatch
) -> None:
    data_provider._write_cached_snapshot(
        "AAPL", _snapshot(), pd.DataFrame(), pd.DataFrame()
    )
    data_provider.SNAPSHOT_MEMORY_CACHE.clear()

    def unexpected_provider_call(_symbol: str):
        raise AssertionError("fresh empty statements should be read from disk")

    monkeypatch.setattr(data_provider.yf, "Ticker", unexpected_provider_call)

    snapshot, income, cashflow = data_provider.get_ticker_snapshot("AAPL")

    assert snapshot.short_name == "Apple"
    assert income.empty
    assert cashflow.empty


@pytest.mark.parametrize("symbol", ["../../outside", "A/B", r"A\\B", "AAPL$"])
def test_cache_paths_reject_unsafe_ticker_symbols(
    isolated_caches, symbol: str
) -> None:
    with pytest.raises(InvalidTickerError):
        data_provider._cache_path_for(symbol)
    with pytest.raises(InvalidTickerError):
        data_provider._snapshot_cache_paths_for(symbol)


@pytest.mark.parametrize("symbol", ["AAPL", "BRK-B", "^GSPC", "BTC-USD", "GC=F"])
def test_cache_paths_accept_supported_yahoo_symbols(isolated_caches, symbol: str) -> None:
    path = data_provider._cache_path_for(symbol)
    assert path.parent.resolve() == data_provider.CACHE_DIR.resolve()
