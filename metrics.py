from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from charts import build_dividend_chart, build_valuation_chart
from data_provider import get_daily_close_history, get_full_price_history, get_price_history, get_ticker_snapshot
from errors import MissingDataError, UnsupportedAnalysisError, to_issue
from models import DividendAnalysisResult, Fundamentals, TickerSnapshot, ValuationAnalysisResult
from portfolio_data import build_holdings_ticker_details


BENCHMARK_TICKER = "SPY"
RETURN_WINDOW_OFFSETS = [
    ("return_1d_pct", pd.DateOffset(days=1), False),
    ("return_1w_pct", pd.DateOffset(weeks=1), False),
    ("return_1m_pct", pd.DateOffset(months=1), False),
    ("return_3m_pct", pd.DateOffset(months=3), False),
    ("return_6m_pct", pd.DateOffset(months=6), False),
    ("return_1y_pct", pd.DateOffset(years=1), True),
    ("return_2y_pct", pd.DateOffset(years=2), True),
    ("return_5y_pct", pd.DateOffset(years=5), True),
    ("return_10y_pct", pd.DateOffset(years=10), True),
]


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _normalize_dividend_yield(value: float | int | None) -> float | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if numeric >= 0.15:
        return numeric / 100
    return numeric


def _history_span_days(history: pd.DataFrame | pd.Series | None) -> int:
    if history is None or len(history.index) < 2:
        return 0
    return int((pd.Timestamp(history.index.max()) - pd.Timestamp(history.index.min())).days)


def _should_show_dividend_bars(dividends_to_plot: pd.DataFrame, history: pd.DataFrame) -> bool:
    if dividends_to_plot.empty:
        return False
    if _history_span_days(history) <= 365 * 2:
        return True
    if len(dividends_to_plot.index) < 2:
        return True
    median_spacing_days = (
        dividends_to_plot.index.to_series().diff().dropna().dt.days.median()
    )
    return pd.isna(median_spacing_days) or median_spacing_days > 45


def _holding_detail_for_ticker(ticker: str) -> dict[str, object] | None:
    try:
        holdings = build_holdings_ticker_details()
    except Exception:
        return None
    if holdings.empty:
        return None
    matches = holdings[holdings["Ticker"].astype(str).str.upper() == ticker.strip().upper()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "quantity": _safe_float(row.get("Quantity")),
        "buy_date": row.get("Buy Date"),
        "bought_at": _safe_float(row.get("Bought at")),
        "market_value": _safe_float(row.get("Market Value")),
        "gain": _safe_float(row.get("Gain")),
        "income": _safe_float(row.get("Income")),
    }


def _compute_return_windows(history: pd.DataFrame | None) -> dict[str, float | None]:
    returns = {field_name: None for field_name, _, _ in RETURN_WINDOW_OFFSETS}
    if history is None or history.empty:
        return returns

    price_column = "Adj Close" if "Adj Close" in history.columns else "Close" if "Close" in history.columns else None
    if price_column is None:
        return returns

    price_series = history[price_column].dropna().copy()
    if len(price_series.index) < 2:
        return returns

    price_series.index = pd.to_datetime(price_series.index).tz_localize(None)
    end_ts = pd.Timestamp(price_series.index.max())
    end_value = _safe_float(price_series.iloc[-1])
    if end_value in (None, 0):
        return returns

    for field_name, offset, annualize in RETURN_WINDOW_OFFSETS:
        start_ts = end_ts - offset
        window = price_series.loc[price_series.index >= start_ts]
        if len(window.index) < 2:
            continue
        start_value = _safe_float(window.iloc[0])
        if start_value in (None, 0):
            continue
        period_return = (end_value / start_value) - 1.0
        if annualize:
            window_days = max((pd.Timestamp(window.index.max()) - pd.Timestamp(window.index.min())).days, 1)
            period_return = (end_value / start_value) ** (365.0 / window_days) - 1.0
        returns[field_name] = period_return * 100.0

    return returns


def _build_fundamentals(
    snapshot: TickerSnapshot,
    history: pd.DataFrame | None,
    *,
    full_history: pd.DataFrame | None = None,
    include_holdings_detail: bool = False,
) -> Fundamentals:
    start_date = None
    end_date = None
    annual_return_pct = None
    annual_return_adj_pct = None
    annual_volatility_pct = None
    alpha_vs_spy_pct = None
    beta_vs_spy = None
    sharpe_ratio = None
    sharpe_ratio_adj = None
    trailing_eps = None
    holdings_detail = _holding_detail_for_ticker(snapshot.ticker) if include_holdings_detail else None
    return_windows = _compute_return_windows(full_history if full_history is not None else history)

    if (
        snapshot.regular_market_price not in (None, 0)
        and snapshot.trailing_pe not in (None, 0)
        and snapshot.trailing_pe is not None
    ):
        trailing_eps = snapshot.regular_market_price / snapshot.trailing_pe

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

        alpha_vs_spy_pct, beta_vs_spy = _compute_benchmark_metrics(history, start_date, end_date)

        if close_volatility not in (None, 0):
            sharpe_ratio = (close_return - 0.03) / close_volatility
        if adj_volatility not in (None, 0):
            sharpe_ratio_adj = (adj_return - 0.03) / adj_volatility

    return Fundamentals(
        ticker=snapshot.ticker,
        name=snapshot.short_name,
        price=snapshot.regular_market_price,
        dividend_yield=_normalize_dividend_yield(snapshot.dividend_yield),
        trailing_pe=snapshot.trailing_pe,
        trailing_eps=trailing_eps,
        asset_type=snapshot.asset_type,
        start_date=start_date,
        end_date=end_date,
        annual_return_pct=annual_return_pct,
        annual_return_adj_pct=annual_return_adj_pct,
        annual_volatility_pct=annual_volatility_pct,
        alpha_vs_spy_pct=alpha_vs_spy_pct,
        beta_vs_spy=beta_vs_spy,
        sharpe_ratio=sharpe_ratio,
        sharpe_ratio_adj=sharpe_ratio_adj,
        holdings_quantity=(holdings_detail or {}).get("quantity"),
        holdings_buy_date=(holdings_detail or {}).get("buy_date"),
        holdings_bought_at=(holdings_detail or {}).get("bought_at"),
        holdings_market_value=(holdings_detail or {}).get("market_value"),
        holdings_gain=(holdings_detail or {}).get("gain"),
        holdings_income=(holdings_detail or {}).get("income"),
        return_1d_pct=return_windows["return_1d_pct"],
        return_1w_pct=return_windows["return_1w_pct"],
        return_1m_pct=return_windows["return_1m_pct"],
        return_3m_pct=return_windows["return_3m_pct"],
        return_6m_pct=return_windows["return_6m_pct"],
        return_1y_pct=return_windows["return_1y_pct"],
        return_2y_pct=return_windows["return_2y_pct"],
        return_5y_pct=return_windows["return_5y_pct"],
        return_10y_pct=return_windows["return_10y_pct"],
    )


def _compute_benchmark_metrics(
    history: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[float | None, float | None]:
    try:
        benchmark_history = get_price_history(BENCHMARK_TICKER, start_date, end_date)
    except Exception:
        return None, None

    if benchmark_history.empty:
        return None, None

    asset_returns = history["Adj Close"].pct_change()
    benchmark_returns = benchmark_history["Adj Close"].pct_change()
    aligned = pd.concat(
        [
            asset_returns.rename("asset"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return None, None

    benchmark_variance = aligned["benchmark"].var()
    if benchmark_variance in (None, 0) or pd.isna(benchmark_variance):
        return None, None

    covariance = aligned["asset"].cov(aligned["benchmark"])
    beta = covariance / benchmark_variance
    alpha_daily = aligned["asset"].mean() - beta * aligned["benchmark"].mean()
    alpha_annual_pct = alpha_daily * 252 * 100
    return _safe_float(alpha_annual_pct), _safe_float(beta)


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


def _payments_per_year(dividend_history: pd.DataFrame) -> int:
    dividends = dividend_history[dividend_history["Dividends"] > 0]
    if dividends.empty:
        return 0

    trailing_year_start = dividends.index.max() - pd.DateOffset(years=1)
    trailing_year_count = int((dividends.index >= trailing_year_start).sum())
    if trailing_year_count > 0:
        return trailing_year_count

    if len(dividends.index) < 2:
        return 1

    median_spacing_days = dividends.index.to_series().diff().dropna().dt.days.median()
    if pd.isna(median_spacing_days):
        return 1
    if median_spacing_days <= 40:
        return 12
    if median_spacing_days <= 120:
        return 4
    if median_spacing_days <= 220:
        return 2
    return 1


def _compute_dividend_labels(dividends_to_plot: pd.DataFrame, dividend_history: pd.DataFrame) -> list[str]:
    if dividends_to_plot.empty:
        return []
    payments_per_year = _payments_per_year(dividend_history)
    if payments_per_year <= 0:
        return []

    return [
        f"{(100 * dividend * payments_per_year) / close_price:.2f}%"
        for dividend, close_price in zip(dividends_to_plot["Dividends"], dividends_to_plot["Close"])
        if close_price not in (None, 0) and not pd.isna(close_price)
    ]


def _build_rebased_benchmark_series(
    reference_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
) -> pd.Series | None:
    if (
        reference_history.empty
        or benchmark_history.empty
        or "Adj Close" not in benchmark_history.columns
        or "Close" not in reference_history.columns
    ):
        return None

    benchmark_series = benchmark_history["Adj Close"].dropna().copy()
    if benchmark_series.empty:
        return None

    benchmark_series.index = pd.to_datetime(benchmark_series.index).tz_localize(None)
    aligned_benchmark = benchmark_series.reindex(pd.DatetimeIndex(reference_history.index)).ffill().bfill()
    if aligned_benchmark.empty or pd.isna(aligned_benchmark.iloc[0]) or aligned_benchmark.iloc[0] == 0:
        return None

    start_value = _safe_float(reference_history["Close"].iloc[0])
    if start_value in (None, 0):
        return None

    return (aligned_benchmark / aligned_benchmark.iloc[0]) * start_value


def analyze_dividend_ticker(ticker: str, start_date: date) -> DividendAnalysisResult:
    try:
        snapshot, _, _ = get_ticker_snapshot(ticker)
        history = get_price_history(ticker, start_date, date.today())
        if history.empty:
            raise MissingDataError(f"No price history is available for {ticker} in the selected date range.")
        full_history = get_full_price_history(ticker)
        benchmark_history = get_full_price_history(BENCHMARK_TICKER)

        fundamentals = _build_fundamentals(
            snapshot,
            history,
            full_history=full_history,
            include_holdings_detail=True,
        )
        plot_history = history.copy()
        plot_history["Adj Close Rebased"] = (
            plot_history["Adj Close"] + (plot_history["Close"].iloc[0] - plot_history["Adj Close"].iloc[0])
        )
        dividends_to_plot = plot_history[plot_history["Dividends"] > 0]
        bar_labels = _compute_dividend_labels(dividends_to_plot, full_history)
        benchmark_series = _build_rebased_benchmark_series(plot_history, benchmark_history)
        figure = build_dividend_chart(
            snapshot,
            plot_history,
            dividends_to_plot,
            bar_labels,
            benchmark_series=benchmark_series,
            show_dividend_bars=_should_show_dividend_bars(dividends_to_plot, plot_history),
        )

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


def _quarterly_series_as_pct_of_price(
    quarterly_series: pd.Series,
    price_series: pd.Series,
    plot_start_date: date,
) -> pd.Series:
    aligned_price = price_series.sort_index().reindex(quarterly_series.index.sort_values(), method="ffill")
    annualized_quarterly_series = quarterly_series * 4
    pct_series = ((annualized_quarterly_series / aligned_price) * 100).replace([np.inf, -np.inf], np.nan).dropna()
    return pct_series.loc[plot_start_date:]


def _compute_free_cash_flow_inputs(
    cashflow_statement: pd.DataFrame,
    snapshot: TickerSnapshot,
    plot_start_date: date,
) -> pd.Series:
    if cashflow_statement.empty:
        raise MissingDataError("Quarterly cash flow statement data is not available for free-cash-flow analysis.")

    if "Free Cash Flow" in cashflow_statement.index:
        free_cash_flow = cashflow_statement.loc["Free Cash Flow"].sort_index()
    else:
        operating_cash_flow = _get_first_matching_row(
            cashflow_statement,
            [
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
                "Total Cash From Operating Activities",
            ],
            "Operating cash flow data is unavailable for free-cash-flow analysis.",
        )
        capital_expenditures = _get_first_matching_row(
            cashflow_statement,
            [
                "Capital Expenditure",
                "Capital Expenditures",
                "Purchase Of PPE",
            ],
            "Capital expenditure data is unavailable for free-cash-flow analysis.",
        )
        if capital_expenditures.dropna().median() <= 0:
            free_cash_flow = operating_cash_flow + capital_expenditures
        else:
            free_cash_flow = operating_cash_flow - capital_expenditures

    if snapshot.shares_outstanding in (None, 0):
        raise MissingDataError("Share count data is unavailable for free-cash-flow analysis.")

    free_cash_flow_per_share = (free_cash_flow / snapshot.shares_outstanding).replace([np.inf, -np.inf], np.nan).dropna()
    if free_cash_flow_per_share.empty:
        raise MissingDataError("Free-cash-flow per share could not be derived for this analysis.")

    return free_cash_flow_per_share.loc[plot_start_date:]


def analyze_valuation_ticker(ticker: str, start_date: date) -> ValuationAnalysisResult:
    try:
        snapshot, quarterly_income_stmt, quarterly_cashflow_stmt = get_ticker_snapshot(ticker)
        history = get_price_history(ticker, start_date, date.today())
        if history.empty:
            raise MissingDataError(f"No price history is available for {ticker} in the selected date range.")
        price_series = get_daily_close_history(ticker, start_date, date.today())
        full_history = get_full_price_history(ticker)
        fundamentals = _build_fundamentals(
            snapshot,
            history,
            full_history=full_history,
            include_holdings_detail=True,
        )
    except Exception as exc:
        return ValuationAnalysisResult(ticker=ticker, issue=to_issue(exc))

    result = ValuationAnalysisResult(ticker=ticker, fundamentals=fundamentals)
    price_plot = None
    pe_plot = None
    eps_pct_plot = None
    revenue_pct_plot = None
    free_cash_flow_pct_plot = None
    show_quarterly_bars = _history_span_days(price_series) <= 365 * 10

    try:
        price_plot, pe_plot, eps_plot = _compute_pe_inputs(
            quarterly_income_stmt,
            snapshot,
            price_series,
            start_date,
        )
        eps_pct_plot = _quarterly_series_as_pct_of_price(eps_plot, price_series, start_date)
    except Exception as exc:
        result.pe_issue = to_issue(exc)

    try:
        _, _, revenue_plot = _compute_ps_inputs(
            quarterly_income_stmt,
            snapshot,
            price_series,
            start_date,
        )
        revenue_pct_plot = _quarterly_series_as_pct_of_price(revenue_plot, price_series, start_date)
    except Exception as exc:
        result.ps_issue = to_issue(exc)

    try:
        free_cash_flow_plot = _compute_free_cash_flow_inputs(
            quarterly_cashflow_stmt,
            snapshot,
            start_date,
        )
        free_cash_flow_pct_plot = _quarterly_series_as_pct_of_price(free_cash_flow_plot, price_series, start_date)
    except Exception as exc:
        result.fcf_issue = to_issue(exc)

    if price_plot is not None and pe_plot is not None:
        result.pe_figure = build_valuation_chart(
            snapshot,
            price_plot,
            pe_plot,
            eps_pct_plot=eps_pct_plot,
            revenue_pct_plot=revenue_pct_plot,
            free_cash_flow_pct_plot=free_cash_flow_pct_plot,
            show_quarterly_bars=show_quarterly_bars,
        )

    return result


def fundamentals_to_frame(
    fundamentals_list: list[Fundamentals],
    include_eps: bool = False,
    include_holdings: bool = False,
    include_return_windows: bool = False,
) -> pd.DataFrame:
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
        ("Alpha vs SPY (%)", "alpha_vs_spy_pct"),
        ("Beta vs SPY", "beta_vs_spy"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sharpe Ratio Adj", "sharpe_ratio_adj"),
    ]
    if include_eps:
        rows.insert(3, ("EPS", "trailing_eps"))
    if include_holdings:
        asset_type_index = next(
            (index for index, (_, field_name) in enumerate(rows) if field_name == "asset_type"),
            len(rows) - 1,
        )
        rows[asset_type_index + 1:asset_type_index + 1] = [
            ("Quantity", "holdings_quantity"),
            ("Buy Date", "holdings_buy_date"),
            ("Bought At", "holdings_bought_at"),
            ("Market Value", "holdings_market_value"),
            ("Gain (%)", "holdings_gain"),
            ("Income", "holdings_income"),
        ]
    if include_return_windows:
        rows.extend(
            [
                ("Return 1D (%)", "return_1d_pct"),
                ("Return 1W (%)", "return_1w_pct"),
                ("Return 1M (%)", "return_1m_pct"),
                ("Return 3M (%)", "return_3m_pct"),
                ("Return 6M (%)", "return_6m_pct"),
                ("Return 1Y (%)", "return_1y_pct"),
                ("Return 2Y (%)", "return_2y_pct"),
                ("Return 5Y (%)", "return_5y_pct"),
                ("Return 10Y (%)", "return_10y_pct"),
            ]
        )

    columns = {}
    for fundamentals in fundamentals_list:
        columns[fundamentals.ticker] = {
            label: (
                getattr(fundamentals, field_name) * 100
                if field_name in {"dividend_yield", "holdings_gain"} and getattr(fundamentals, field_name) is not None
                else getattr(fundamentals, field_name)
            )
            for label, field_name in rows
        }

    return pd.DataFrame(columns)
