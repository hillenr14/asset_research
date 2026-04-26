from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from charts import build_dividend_chart, build_pe_chart, build_ps_chart
from data_provider import get_daily_close_history, get_price_history, get_ticker_snapshot
from errors import MissingDataError, UnsupportedAnalysisError, to_issue
from models import DividendAnalysisResult, Fundamentals, TickerSnapshot, ValuationAnalysisResult


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _build_fundamentals(snapshot: TickerSnapshot, history: pd.DataFrame | None) -> Fundamentals:
    start_date = None
    end_date = None
    annual_return_pct = None
    annual_return_adj_pct = None
    annual_volatility_pct = None
    sharpe_ratio = None
    sharpe_ratio_adj = None

    if history is not None and not history.empty:
        start_date = pd.Timestamp(history.index.min()).date()
        end_date = pd.Timestamp(history.index.max()).date()

        returns = history[["Close", "Adj Close"]].pct_change().fillna(0.0)
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)

        annual_return_pct = _safe_float(annual_return.get("Close")) * 100 if _safe_float(annual_return.get("Close")) is not None else None
        annual_return_adj_pct = _safe_float(annual_return.get("Adj Close")) * 100 if _safe_float(annual_return.get("Adj Close")) is not None else None
        annual_volatility_pct = _safe_float(annual_volatility.get("Close")) * 100 if _safe_float(annual_volatility.get("Close")) is not None else None

        close_volatility = _safe_float(annual_volatility.get("Close"))
        adj_volatility = _safe_float(annual_volatility.get("Adj Close"))
        close_return = _safe_float(annual_return.get("Close"))
        adj_return = _safe_float(annual_return.get("Adj Close"))

        if close_volatility not in (None, 0):
            sharpe_ratio = (close_return - 0.03) / close_volatility
        if adj_volatility not in (None, 0):
            sharpe_ratio_adj = (adj_return - 0.03) / adj_volatility

    return Fundamentals(
        ticker=snapshot.ticker,
        name=snapshot.short_name,
        price=snapshot.regular_market_price,
        dividend_yield=snapshot.dividend_yield,
        trailing_pe=snapshot.trailing_pe,
        asset_type=snapshot.asset_type,
        start_date=start_date,
        end_date=end_date,
        annual_return_pct=annual_return_pct,
        annual_return_adj_pct=annual_return_adj_pct,
        annual_volatility_pct=annual_volatility_pct,
        sharpe_ratio=sharpe_ratio,
        sharpe_ratio_adj=sharpe_ratio_adj,
    )


def _get_first_matching_row(
    statement: pd.DataFrame,
    row_names: list[str],
    error_message: str,
) -> pd.Series:
    for row_name in row_names:
        if row_name in statement.index:
            return statement.loc[row_name].sort_index()
    raise UnsupportedAnalysisError(error_message)


def _build_share_series(statement: pd.DataFrame, snapshot: TickerSnapshot, reference_index: pd.Index) -> pd.Series:
    for row_name in ["Diluted Average Shares", "Basic Average Shares"]:
        if row_name in statement.index:
            return statement.loc[row_name].sort_index()

    if snapshot.shares_outstanding in (None, 0):
        raise MissingDataError("Share count data is unavailable for this analysis.")

    return pd.Series(snapshot.shares_outstanding, index=reference_index)


def _compute_dividend_labels(dividends_to_plot: pd.DataFrame, history: pd.DataFrame) -> list[str]:
    dividend_count_1y = len(
        history[
            (history.index >= history.index.max() - pd.DateOffset(years=1))
            & (history["Dividends"] > 0)
        ]
    )
    if dividend_count_1y == 0:
        return []

    return [
        f"{(dividend * 100 * dividend_count_1y) / close_price:.2f}%"
        for dividend, close_price in zip(dividends_to_plot["Dividends"], dividends_to_plot["Close"])
    ]


def analyze_dividend_ticker(ticker: str, start_date: date) -> DividendAnalysisResult:
    try:
        snapshot, _ = get_ticker_snapshot(ticker)
        history = get_price_history(ticker, start_date, date.today())
        if history.empty:
            raise MissingDataError(f"No price history is available for {ticker} in the selected date range.")

        fundamentals = _build_fundamentals(snapshot, history)
        plot_history = history.copy()
        plot_history["Adj Close Rebased"] = (
            plot_history["Adj Close"] + (plot_history["Close"].iloc[0] - plot_history["Adj Close"].iloc[0])
        )
        dividends_to_plot = plot_history[plot_history["Dividends"] > 0]
        bar_labels = _compute_dividend_labels(dividends_to_plot, plot_history)
        figure = build_dividend_chart(snapshot, plot_history, dividends_to_plot, bar_labels)

        return DividendAnalysisResult(
            ticker=ticker,
            fundamentals=fundamentals,
            history=history,
            figure=figure,
        )
    except Exception as exc:
        return DividendAnalysisResult(ticker=ticker, issue=to_issue(exc))


def _compute_pe_inputs(
    statement: pd.DataFrame,
    snapshot: TickerSnapshot,
    price_series: pd.Series,
    plot_start_date: date,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if statement.empty:
        raise MissingDataError("Quarterly income statement data is not available for P/E analysis.")

    net_income = _get_first_matching_row(
        statement,
        ["Net Income", "Net Income Common Stockholders"],
        "Net income data is unavailable for P/E analysis.",
    )
    shares = _build_share_series(statement, snapshot, net_income.index).replace(0, np.nan)

    eps_quarterly = (net_income / shares).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if eps_quarterly.empty:
        raise MissingDataError("Quarterly EPS could not be derived for P/E analysis.")

    eps_ttm = eps_quarterly.rolling(window=4, min_periods=1).mean() * 4
    eps_ttm_daily = eps_ttm.reindex(price_series.index, method="ffill")
    pe_series = (price_series / eps_ttm_daily).replace([np.inf, -np.inf], np.nan).dropna()
    if pe_series.empty:
        raise MissingDataError("Historical P/E series could not be computed.")

    return (
        price_series.loc[plot_start_date:],
        pe_series.loc[plot_start_date:],
        eps_quarterly.loc[plot_start_date:],
    )


def _compute_ps_inputs(
    statement: pd.DataFrame,
    snapshot: TickerSnapshot,
    price_series: pd.Series,
    plot_start_date: date,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if statement.empty:
        raise MissingDataError("Quarterly income statement data is not available for P/S analysis.")

    revenue = _get_first_matching_row(
        statement,
        ["Total Revenue", "Revenue"],
        "Revenue data is unavailable for P/S analysis.",
    )
    if snapshot.shares_outstanding in (None, 0):
        raise MissingDataError("Share count data is unavailable for P/S analysis.")

    revenue_per_share_quarterly = (revenue / snapshot.shares_outstanding).dropna().sort_index()
    if revenue_per_share_quarterly.empty:
        raise MissingDataError("Revenue-per-share data could not be derived for P/S analysis.")

    revenue_ttm = revenue_per_share_quarterly.rolling(window=4, min_periods=1).mean() * 4
    revenue_ttm_daily = revenue_ttm.reindex(price_series.index, method="ffill")
    ps_series = (price_series / revenue_ttm_daily).replace([np.inf, -np.inf], np.nan).dropna()
    if ps_series.empty:
        raise MissingDataError("Historical P/S series could not be computed.")

    return (
        price_series.loc[plot_start_date:],
        ps_series.loc[plot_start_date:],
        revenue_per_share_quarterly.loc[plot_start_date:],
    )


def analyze_valuation_ticker(ticker: str, start_date: date) -> ValuationAnalysisResult:
    try:
        snapshot, quarterly_income_stmt = get_ticker_snapshot(ticker)
        history = get_price_history(ticker, start_date, date.today())
        if history.empty:
            raise MissingDataError(f"No price history is available for {ticker} in the selected date range.")
        price_series = get_daily_close_history(ticker, start_date, date.today())
        fundamentals = _build_fundamentals(snapshot, history)
    except Exception as exc:
        return ValuationAnalysisResult(ticker=ticker, issue=to_issue(exc))

    result = ValuationAnalysisResult(ticker=ticker, fundamentals=fundamentals)

    try:
        price_plot, pe_plot, eps_plot = _compute_pe_inputs(
            quarterly_income_stmt,
            snapshot,
            price_series,
            start_date,
        )
        result.pe_figure = build_pe_chart(snapshot, price_plot, pe_plot, eps_plot)
    except Exception as exc:
        result.pe_issue = to_issue(exc)

    try:
        price_plot, ps_plot, revenue_plot = _compute_ps_inputs(
            quarterly_income_stmt,
            snapshot,
            price_series,
            start_date,
        )
        result.ps_figure = build_ps_chart(snapshot, price_plot, ps_plot, revenue_plot)
    except Exception as exc:
        result.ps_issue = to_issue(exc)

    return result


def fundamentals_to_frame(fundamentals_list: list[Fundamentals]) -> pd.DataFrame:
    if not fundamentals_list:
        return pd.DataFrame()

    rows = [
        ("Price", "price"),
        ("Dividend Yield (%)", "dividend_yield"),
        ("Trailing P/E", "trailing_pe"),
        ("Asset Type", "asset_type"),
        ("Annual Return (%)", "annual_return_pct"),
        ("Annual Return Adj (%)", "annual_return_adj_pct"),
        ("Annual Volatility (%)", "annual_volatility_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sharpe Ratio Adj", "sharpe_ratio_adj"),
    ]

    columns = {}
    for fundamentals in fundamentals_list:
        columns[fundamentals.ticker] = {
            label: getattr(fundamentals, field_name)
            for label, field_name in rows
        }

    return pd.DataFrame(columns)
