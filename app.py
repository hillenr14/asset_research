from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from charts import build_holdings_portfolio_chart
from config import (
    DEFAULT_LOOKBACK,
    LOOKBACK_OPTIONS,
    load_config,
    lookback_exceeds_years,
    lookback_start_date,
    rolling_analysis_start_date,
    save_config,
)
from data_provider import clear_in_memory_price_history_cache, get_full_price_history, get_ticker_snapshot, warm_price_history_cache
from portfolio_data import (
    HOLDINGS_NAVIGABLE_TYPES,
    build_holdings_analysis_table,
    build_holdings_income_by_month_table,
    build_holdings_portfolio_histories,
    build_holdings_portfolio_window,
    compute_holdings_totals,
    format_income_by_month_for_display,
    format_portfolio_holdings_for_display,
    holdings_navigation_items,
    portfolio_history_tickers,
    refresh_holdings_analysis_data,
)
from metrics import BENCHMARK_TICKER, analyze_dividend_ticker, analyze_valuation_ticker, fundamentals_to_frame
from models import AnalysisSettings, AppConfig, DividendAnalysisResult, ValuationAnalysisResult
from ui_helpers import (
    build_analysis_summary_frame,
    dataframe_to_csv_bytes,
    figure_to_html,
    format_dataframe_for_display,
    normalize_single_ticker,
    validate_single_ticker,
)


st.set_page_config(
    page_title="Stock Analysis App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_page_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-kicker {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            color: #cbd5e1;
            max-width: 52rem;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stTabs"] button {
            font-weight: 500;
        }

        div[data-testid="stButton"] > button {
            border-radius: 0.85rem;
        }

        div[data-testid="stButton"] > button[kind="secondary"] {
            background: #18263a;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 0.85rem;
        }

        .pane-title {
            font-size: 0.95rem;
            color: #cbd5e1;
            margin-bottom: 0.5rem;
            letter-spacing: 0.02em;
        }

        .pane-copy {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .asset-name {
            color: #94a3b8;
            font-size: 0.82rem;
            line-height: 1.2;
            padding-top: 0.35rem;
        }

        .totals-row {
            display: flex;
            gap: 2rem;
            margin-top: 0.9rem;
            color: #cbd5e1;
            font-size: 0.9rem;
        }

        .totals-row strong {
            color: #f8fafc;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    st.markdown("##### :material/query_stats: Stock research dashboard")
    st.markdown(
        '<div class="app-subtitle">Analyze dividend, valuation, and holdings data with full-history local caching and a global lookback control.</div>',
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    config = load_config()
    if "app_config" not in st.session_state:
        st.session_state.app_config = config
    else:
        st.session_state.app_config.dividend_analysis.start_date = analysis_start_date()
        st.session_state.app_config.valuation_analysis.start_date = analysis_start_date()
    st.session_state.setdefault("dividend_results", {})
    st.session_state.setdefault("valuation_results", {})
    active_config = st.session_state.app_config
    st.session_state.setdefault("dividend_selected_ticker", active_config.dividend_analysis.tickers[0] if active_config.dividend_analysis.tickers else None)
    st.session_state.setdefault("valuation_selected_ticker", active_config.valuation_analysis.tickers[0] if active_config.valuation_analysis.tickers else None)
    st.session_state.setdefault("dividend_add_ticker", "")
    st.session_state.setdefault("valuation_add_ticker", "")
    st.session_state.setdefault("dividend_show_summary", True)
    st.session_state.setdefault("valuation_show_summary", True)
    st.session_state.setdefault("selected_lookback", DEFAULT_LOOKBACK)
    st.session_state.setdefault("selected_page", "Dividend Analysis")
    st.session_state.setdefault("holdings_show_summary", True)
    st.session_state.setdefault("holdings_selected_ticker", None)
    st.session_state.setdefault("holdings_selected_type", None)
    st.session_state.setdefault("holdings_detail_signature", None)
    st.session_state["show_export_actions"] = False
    st.session_state.setdefault("dividend_auto_signature", None)
    st.session_state.setdefault("valuation_auto_signature", None)
    st.session_state.setdefault("holdings_refreshed_on_load", False)
    st.session_state.setdefault("market_data_warmed_on_load", False)


def current_config() -> AppConfig:
    return st.session_state.app_config


def persist_current_config() -> None:
    save_config(st.session_state.app_config)


def analysis_start_date():
    return lookback_start_date(current_lookback())


def current_lookback() -> str:
    return st.session_state.get("selected_lookback", DEFAULT_LOOKBACK)


def all_market_history_tickers() -> list[str]:
    config = current_config()
    tickers = (
        config.dividend_analysis.tickers
        + config.valuation_analysis.tickers
        + portfolio_history_tickers()
        + [BENCHMARK_TICKER]
    )
    ordered_tickers: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if ticker is None or pd.isna(ticker):
            continue
        symbol = str(ticker).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered_tickers.append(symbol)
    return ordered_tickers


def warm_market_data_on_load() -> None:
    if st.session_state.get("market_data_warmed_on_load", False):
        return

    refresh_holdings_analysis_data()
    clear_in_memory_price_history_cache()
    tickers = all_market_history_tickers()
    if tickers:
        with st.spinner("Refreshing market data cache and loading full histories..."):
            warm_price_history_cache(tickers)
    st.session_state["market_data_warmed_on_load"] = True
    st.session_state["holdings_refreshed_on_load"] = True


def slice_history_for_current_lookback(history: object):
    if history is None or getattr(history, "empty", True):
        return history
    start_ts = pd.Timestamp(analysis_start_date())
    return history.loc[history.index >= start_ts].copy()


def dataframe_height(row_count: int, visible_rows: int | None = None) -> int:
    rows = row_count if visible_rows is None else min(row_count, visible_rows)
    return max(140, 40 + rows * 35)


def fundamentals_table_display(detail_df: pd.DataFrame) -> pd.DataFrame:
    currency_metrics = {"Price", "EPS", "Bought At", "Market Value", "Income"}
    percent_metrics = {
        "Dividend Yield (%)",
        "Gain (%)",
        "Annual Return (%)",
        "Annual Return Adj (%)",
        "Annual Volatility (%)",
        "Alpha vs SPY (%)",
        "Return 1D (%)",
        "Return 1W (%)",
        "Return 1M (%)",
        "Return 3M (%)",
        "Return 6M (%)",
        "Return 1Y (%)",
        "Return 2Y (%)",
        "Return 5Y (%)",
        "Return 10Y (%)",
    }
    integer_metrics = {"Quantity"}
    date_metrics = {"Buy Date"}

    display_df = detail_df.copy().astype("object")
    for metric in display_df.index:
        for column in display_df.columns:
            value = display_df.at[metric, column]
            if value is None or pd.isna(value):
                display_df.at[metric, column] = "N/A"
            elif metric in currency_metrics:
                display_df.at[metric, column] = f"${float(value):,.2f}"
            elif metric in percent_metrics:
                display_df.at[metric, column] = f"{float(value):.2f}%"
            elif metric in integer_metrics:
                display_df.at[metric, column] = f"{round(float(value)):,}"
            elif metric in date_metrics and hasattr(value, "isoformat"):
                display_df.at[metric, column] = value.isoformat()
            elif isinstance(value, float):
                display_df.at[metric, column] = f"{value:.2f}"
            else:
                display_df.at[metric, column] = str(value)
    return display_df.reset_index().rename(columns={"index": "Metric"})


def rebase_reinvested_series_for_display(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "Reinvested Portfolio Value" not in history.columns:
        return history
    rebased = history.copy()
    start_value = rebased["Portfolio Value"].iloc[0]
    reinvested_start = rebased["Reinvested Portfolio Value"].iloc[0]
    if pd.isna(start_value) or pd.isna(reinvested_start) or reinvested_start in (None, 0):
        return rebased
    scale_factor = start_value / reinvested_start
    rebased["Reinvested Portfolio Value"] = (
        rebased["Reinvested Portfolio Value"] * scale_factor
    )
    return rebased


def build_rebased_benchmark_series(
    benchmark_ticker: str,
    reference_history: pd.DataFrame,
) -> pd.Series | None:
    if reference_history.empty or "Portfolio Value" not in reference_history.columns:
        return None
    try:
        benchmark_history = get_full_price_history(benchmark_ticker)
    except Exception:
        return None
    if benchmark_history.empty or "Adj Close" not in benchmark_history.columns:
        return None

    benchmark_series = benchmark_history["Adj Close"].dropna()
    if benchmark_series.empty:
        return None
    benchmark_series.index = pd.to_datetime(benchmark_series.index).tz_localize(None)
    rebased_index = pd.DatetimeIndex(reference_history.index)
    benchmark_series = benchmark_series.reindex(rebased_index).ffill().bfill()
    if benchmark_series.empty or pd.isna(benchmark_series.iloc[0]) or benchmark_series.iloc[0] == 0:
        return None

    start_value = reference_history["Portfolio Value"].iloc[0]
    return (benchmark_series / benchmark_series.iloc[0]) * start_value


def _annualized_return_from_values(start_value: float, end_value: float, num_days: int) -> float | None:
    if start_value <= 0 or end_value <= 0 or num_days <= 0:
        return None
    return (end_value / start_value) ** (365.0 / num_days) - 1.0


def _annualized_income_return(income_total: float, start_value: float, num_days: int) -> float | None:
    if start_value <= 0 or num_days <= 0:
        return None
    return (income_total / start_value) * (365.0 / num_days)


def _compute_alpha_beta_vs_spy(
    portfolio_series: pd.Series,
    benchmark_series: pd.Series,
) -> tuple[float | None, float | None]:
    portfolio_returns = portfolio_series.pct_change()
    benchmark_returns = benchmark_series.pct_change()
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return None, None
    benchmark_variance = aligned["benchmark"].var()
    if benchmark_variance in (None, 0) or pd.isna(benchmark_variance):
        return None, None
    beta = aligned["portfolio"].cov(aligned["benchmark"]) / benchmark_variance
    alpha_daily = aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()
    alpha_annual = alpha_daily * 252
    return float(alpha_annual), float(beta)


def build_holdings_window_metrics_table(
    portfolio_value_history: pd.DataFrame,
    monthly_income_history: pd.DataFrame,
) -> pd.DataFrame:
    if portfolio_value_history.empty or monthly_income_history.empty:
        return pd.DataFrame()

    window_specs = [
        ("1wk", pd.DateOffset(weeks=1)),
        ("1m", pd.DateOffset(months=1)),
        ("3m", pd.DateOffset(months=3)),
        ("6m", pd.DateOffset(months=6)),
        ("1y", pd.DateOffset(years=1)),
        ("2y", pd.DateOffset(years=2)),
        ("5y", pd.DateOffset(years=5)),
    ]
    benchmark_history = get_full_price_history(BENCHMARK_TICKER)
    benchmark_adj_close = benchmark_history["Adj Close"].dropna() if not benchmark_history.empty and "Adj Close" in benchmark_history.columns else pd.Series(dtype="float64")
    if not benchmark_adj_close.empty:
        benchmark_adj_close.index = pd.to_datetime(benchmark_adj_close.index).tz_localize(None)

    end_ts = pd.Timestamp(portfolio_value_history.index.max())
    columns: dict[str, dict[str, float | None]] = {}

    for label, offset in window_specs:
        start_ts = end_ts - offset
        window_values = portfolio_value_history.loc[portfolio_value_history.index >= start_ts].copy()
        window_income = monthly_income_history.loc[monthly_income_history.index >= start_ts].copy()
        if window_values.empty or len(window_values.index) < 2:
            columns[label] = {metric: None for metric in [
                "Returns (annualized)",
                "Income (annualized)",
                "Total return (annualized)",
                "Volatility (annualized)",
                "Sharpe Ratio",
                "Beta (vs SPY)",
                "Alpha (vs SPY)",
            ]}
            continue

        window_days = max((pd.Timestamp(window_values.index.max()) - pd.Timestamp(window_values.index.min())).days, 1)
        start_value = float(window_values["Portfolio Value"].iloc[0])
        end_value = float(window_values["Portfolio Value"].iloc[-1])
        start_reinvested = float(window_values["Reinvested Portfolio Value"].iloc[0])
        end_reinvested = float(window_values["Reinvested Portfolio Value"].iloc[-1])
        income_total = float(window_income["Income"].sum()) if "Income" in window_income.columns else 0.0

        reinvested_returns = window_values["Reinvested Portfolio Value"].pct_change().dropna()
        volatility = float(reinvested_returns.std() * np.sqrt(252)) if len(reinvested_returns) >= 2 else None
        total_return = _annualized_return_from_values(start_reinvested, end_reinvested, window_days)
        sharpe_ratio = ((total_return - 0.03) / volatility) if total_return is not None and volatility not in (None, 0) else None

        benchmark_slice = pd.Series(dtype="float64")
        if not benchmark_adj_close.empty:
            benchmark_slice = benchmark_adj_close.reindex(pd.DatetimeIndex(window_values.index)).ffill().bfill()
        alpha, beta = (None, None)
        if not benchmark_slice.empty:
            alpha, beta = _compute_alpha_beta_vs_spy(window_values["Reinvested Portfolio Value"], benchmark_slice)

        columns[label] = {
            "Returns (annualized)": _annualized_return_from_values(start_value, end_value, window_days),
            "Income (annualized)": _annualized_income_return(income_total, start_value, window_days),
            "Total return (annualized)": total_return,
            "Volatility (annualized)": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Beta (vs SPY)": beta,
            "Alpha (vs SPY)": alpha,
        }

    metric_order = [
        "Returns (annualized)",
        "Income (annualized)",
        "Total return (annualized)",
        "Volatility (annualized)",
        "Sharpe Ratio",
        "Beta (vs SPY)",
        "Alpha (vs SPY)",
    ]
    return pd.DataFrame(columns).reindex(metric_order)


def format_holdings_window_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    display_df = metrics_df.copy().astype("object")
    percent_rows = {
        "Returns (annualized)",
        "Income (annualized)",
        "Total return (annualized)",
        "Volatility (annualized)",
        "Alpha (vs SPY)",
    }
    decimal_rows = {"Sharpe Ratio", "Beta (vs SPY)"}
    for row_label in display_df.index:
        for column in display_df.columns:
            value = display_df.at[row_label, column]
            if value is None or pd.isna(value):
                display_df.at[row_label, column] = "N/A"
            elif row_label in percent_rows:
                display_df.at[row_label, column] = f"{float(value) * 100:.2f}%"
            elif row_label in decimal_rows:
                display_df.at[row_label, column] = f"{float(value):.2f}"
            else:
                display_df.at[row_label, column] = str(value)
    return display_df.reset_index().rename(columns={"index": "Metric"})


def get_asset_label(ticker: str) -> str:
    try:
        snapshot, _, _ = get_ticker_snapshot(ticker)
        if snapshot.short_name and snapshot.short_name != ticker:
            return f"{ticker} · {snapshot.short_name}"
    except Exception:
        pass
    return ticker


def get_asset_name(ticker: str) -> str:
    try:
        snapshot, _, _ = get_ticker_snapshot(ticker)
        if snapshot.short_name and snapshot.short_name != ticker:
            return snapshot.short_name
    except Exception:
        pass
    return ""


def holdings_analysis_mode(asset_type: str | None) -> str | None:
    normalized_type = str(asset_type or "").strip().lower()
    if normalized_type == "stock":
        return "valuation"
    if normalized_type in {"income", "growth"}:
        return "dividend"
    return None


def switch_holdings_detail(ticker: str, asset_type: str | None) -> None:
    mode = holdings_analysis_mode(asset_type)
    if mode is None:
        return
    st.session_state["holdings_selected_ticker"] = ticker
    st.session_state["holdings_selected_type"] = asset_type
    st.session_state["holdings_show_summary"] = False
    analyze_single_ticker(mode, ticker)
    st.session_state["holdings_detail_signature"] = (ticker, str(asset_type).strip().lower(), current_lookback())


def ensure_holdings_selection() -> None:
    nav_items = holdings_navigation_items()
    selected_ticker = st.session_state.get("holdings_selected_ticker")
    if nav_items.empty:
        st.session_state["holdings_selected_ticker"] = None
        st.session_state["holdings_selected_type"] = None
        return
    if selected_ticker in nav_items["Ticker"].tolist():
        selected_row = nav_items.loc[nav_items["Ticker"] == selected_ticker].iloc[0]
        st.session_state["holdings_selected_type"] = selected_row["Type"]
        return
    first_row = nav_items.iloc[0]
    st.session_state["holdings_selected_ticker"] = first_row["Ticker"]
    st.session_state["holdings_selected_type"] = first_row["Type"]


def sync_holdings_detail() -> None:
    if st.session_state.get("holdings_show_summary", True):
        return
    ticker = st.session_state.get("holdings_selected_ticker")
    asset_type = st.session_state.get("holdings_selected_type")
    mode = holdings_analysis_mode(asset_type)
    if not ticker or mode is None:
        return
    signature = (ticker, str(asset_type).strip().lower(), current_lookback())
    if st.session_state.get("holdings_detail_signature") == signature:
        return
    analyze_single_ticker(mode, ticker)
    st.session_state["holdings_detail_signature"] = signature


def prune_results_for_mode(mode: str, tickers: list[str]) -> None:
    key = f"{mode}_results"
    current_results = st.session_state[key]
    st.session_state[key] = {ticker: result for ticker, result in current_results.items() if ticker in tickers}


def ensure_ticker_selection(mode: str) -> None:
    tickers = getattr(current_config(), f"{mode}_analysis").tickers
    selected_key = f"{mode}_selected_ticker"
    selected = st.session_state.get(selected_key)
    if selected in tickers:
        return
    st.session_state[selected_key] = tickers[0] if tickers else None


def add_ticker(mode: str) -> None:
    input_key = f"{mode}_add_ticker"
    ticker = normalize_single_ticker(st.session_state.get(input_key, ""))
    errors = validate_single_ticker(ticker)
    if errors:
        for message in errors:
            st.warning(message)
        return

    config = current_config()
    settings = getattr(config, f"{mode}_analysis")
    if ticker in settings.tickers:
        st.info(f"{ticker} is already in the list.")
        return

    settings.tickers.append(ticker)
    settings.start_date = analysis_start_date()
    persist_current_config()
    st.session_state[input_key] = ""
    st.session_state[f"{mode}_selected_ticker"] = ticker
    st.session_state[f"{mode}_show_summary"] = False
    analyze_single_ticker(mode, ticker)


def remove_ticker(mode: str, ticker: str) -> None:
    config = current_config()
    settings = getattr(config, f"{mode}_analysis")
    settings.tickers = [value for value in settings.tickers if value != ticker]
    settings.start_date = analysis_start_date()
    persist_current_config()

    result_key = f"{mode}_results"
    st.session_state[result_key].pop(ticker, None)
    st.session_state[f"{mode}_show_summary"] = False
    ensure_ticker_selection(mode)


def analyze_single_ticker(mode: str, ticker: str) -> None:
    if not ticker:
        return

    start_date = analysis_start_date()
    analyzer: Callable[[str, object], object]
    result_key: str
    label: str
    if mode == "dividend":
        analyzer = analyze_dividend_ticker
        result_key = "dividend_results"
        label = "dividend"
    else:
        analyzer = analyze_valuation_ticker
        result_key = "valuation_results"
        label = "valuation"

    with st.spinner(f"Analyzing {label} data for {ticker}..."):
        st.session_state[result_key][ticker] = analyzer(ticker, start_date)


def analyze_all_tickers(mode: str) -> None:
    config = current_config()
    tickers = getattr(config, f"{mode}_analysis").tickers
    if not tickers:
        st.warning("Add at least one ticker first.")
        return

    progress = st.progress(0.0, text=f"Preparing {mode} analysis...")
    for index, ticker in enumerate(tickers, start=1):
        progress.progress(index / len(tickers), text=f"Analyzing {ticker} ({index}/{len(tickers)})")
        analyze_single_ticker(mode, ticker)
    progress.empty()


def sync_initial_analyses(mode: str) -> None:
    tickers = getattr(current_config(), f"{mode}_analysis").tickers
    signature = (tuple(tickers), current_lookback())
    signature_key = f"{mode}_auto_signature"
    if st.session_state.get(signature_key) == signature:
        return
    if tickers:
        analyze_all_tickers(mode)
    st.session_state[signature_key] = signature


def render_export_controls(label_prefix: str, csv_name: str | None = None, csv_data: bytes | None = None, html_name: str | None = None, html_data: str | None = None) -> None:
    if not st.session_state.get("show_export_actions", False):
        return

    columns = st.columns(2)
    with columns[0]:
        if csv_name and csv_data is not None:
            st.download_button(
                "Download table (CSV)",
                data=csv_data,
                file_name=csv_name,
                mime="text/csv",
                key=f"{label_prefix}-csv",
            )
    with columns[1]:
        if html_name and html_data is not None:
            st.download_button(
                "Download chart (HTML)",
                data=html_data,
                file_name=html_name,
                mime="text/html",
                key=f"{label_prefix}-html",
            )


def render_asset_list(mode: str) -> None:
    config = current_config()
    settings = getattr(config, f"{mode}_analysis")
    settings.start_date = analysis_start_date()

    pane_title = "Dividend Assets" if mode == "dividend" else "Valuation Assets"
    st.markdown(f'<div class="pane-title">{pane_title}</div>', unsafe_allow_html=True)
    st.caption(f"Lookback: {current_lookback()}")

    if st.button("Summary", key=f"{mode}-summary-button", width="stretch"):
        st.session_state[f"{mode}_show_summary"] = True
        st.rerun()

    add_cols = st.columns([1.1, 4.4])
    with add_cols[0]:
        if st.button("➕", key=f"{mode}-add-button", width="stretch"):
            add_ticker(mode)
            st.rerun()
    with add_cols[1]:
        st.text_input(
            "Add ticker",
            key=f"{mode}_add_ticker",
            label_visibility="collapsed",
            placeholder="Enter ticker",
        )

    st.divider()

    for ticker in settings.tickers:
        row = st.columns([0.8, 1.8, 3.4])
        with row[0]:
            if st.button("✖️", key=f"{mode}-remove-{ticker}", width="stretch"):
                remove_ticker(mode, ticker)
                st.rerun()
        with row[1]:
            if st.button(
                ticker,
                key=f"{mode}-select-{ticker}",
                width="stretch",
                type="primary" if st.session_state.get(f"{mode}_selected_ticker") == ticker else "secondary",
            ):
                st.session_state[f"{mode}_selected_ticker"] = ticker
                st.session_state[f"{mode}_show_summary"] = False
                analyze_single_ticker(mode, ticker)
                st.rerun()
        with row[2]:
            asset_name = get_asset_name(ticker)
            if asset_name:
                st.markdown(f'<div class="asset-name">{asset_name}</div>', unsafe_allow_html=True)


def interactive_table_display(
    source_df: pd.DataFrame,
    display_df: pd.DataFrame,
    key_prefix: str,
    visible_rows: int = 20,
    hidden_columns: set[str] | None = None,
) -> None:
    if source_df.empty or display_df.empty:
        st.info("No data available.")
        return

    hidden_columns = hidden_columns or set()
    visible_columns = [column for column in display_df.columns if column not in hidden_columns]
    widths = [0.55] + [
        1.35 if column in {"Ticker", "Asset", "Description", "Total (1Y)"} else 1.0
        for column in visible_columns
    ]

    header = st.columns(widths)
    with header[0]:
        st.markdown(" ")
    for index, column in enumerate(visible_columns, start=1):
        with header[index]:
            st.markdown(f"**{column}**")

    rows_to_render = min(len(display_df), visible_rows)
    for row_index in range(rows_to_render):
        source_row = source_df.iloc[row_index]
        display_row = display_df.iloc[row_index]
        row_cols = st.columns(widths)
        with row_cols[0]:
            ticker = str(source_row.get("Ticker", "")).strip().upper()
            asset_type = source_row.get("Type")
            if holdings_analysis_mode(asset_type) is not None and ticker:
                if st.button(
                    "↗",
                    key=f"{key_prefix}-open-{row_index}-{ticker}",
                    width="stretch",
                ):
                    switch_holdings_detail(ticker, asset_type)
                    st.rerun()
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
        for index, column in enumerate(visible_columns, start=1):
            with row_cols[index]:
                st.markdown(str(display_row.get(column, "")))


def render_holdings_asset_list() -> None:
    st.markdown('<div class="pane-title">Holdings Assets</div>', unsafe_allow_html=True)
    st.caption(f"Lookback: {current_lookback()}")

    if st.button("Summary", key="holdings-summary-button", width="stretch"):
        st.session_state["holdings_show_summary"] = True
        st.rerun()

    st.divider()
    nav_items = holdings_navigation_items()
    for _, row in nav_items.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        asset_type = row["Type"]
        row_cols = st.columns([1.8, 3.4])
        with row_cols[0]:
            if st.button(
                ticker,
                key=f"holdings-select-{ticker}",
                width="stretch",
                type="primary" if st.session_state.get("holdings_selected_ticker") == ticker and not st.session_state.get("holdings_show_summary", True) else "secondary",
            ):
                switch_holdings_detail(ticker, asset_type)
                st.rerun()
        with row_cols[1]:
            asset_name = row.get("Description") or get_asset_name(ticker)
            if asset_name:
                st.markdown(f'<div class="asset-name">{asset_name}</div>', unsafe_allow_html=True)


def render_dividend_main(
    result: DividendAnalysisResult | None,
    ticker: str | None,
    *,
    include_holdings_context: bool = False,
) -> None:
    if not ticker:
        st.info("Add a ticker on the left to view dividend analysis.")
        return

    if result is None:
        st.info("Click an asset on the left to analyze and display it.")
        return

    st.header(f"Dividend Analysis: {ticker}")
    if result.issue:
        st.error(result.issue.message)
        if result.issue.details:
            st.caption(result.issue.details)
        return

    if not result.fundamentals or result.figure is None:
        st.warning("No dividend output was generated for this ticker.")
        return

    st.markdown('<div class="pane-copy">Selected asset output appears here after you click a ticker in the left pane.</div>', unsafe_allow_html=True)
    detail_df = fundamentals_to_frame(
        [result.fundamentals],
        include_holdings=include_holdings_context,
        include_return_windows=include_holdings_context,
    )
    display_df = fundamentals_table_display(detail_df)
    left, right = st.columns([0.9, 2.8], vertical_alignment="top")
    with left:
        st.dataframe(
            display_df,
            width="stretch",
            height=dataframe_height(len(display_df)),
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                ticker: st.column_config.TextColumn(ticker, width="small"),
            },
        )
        render_export_controls(
            label_prefix=f"{ticker}-dividend",
            csv_name=f"{ticker.lower()}_dividend_fundamentals.csv",
            csv_data=dataframe_to_csv_bytes(detail_df),
        )
    with right:
        st.plotly_chart(result.figure, width="stretch")
        render_export_controls(
            label_prefix=f"{ticker}-dividend-chart",
            html_name=f"{ticker.lower()}_dividend_chart.html",
            html_data=figure_to_html(result.figure),
        )


def render_valuation_main(
    result: ValuationAnalysisResult | None,
    ticker: str | None,
    *,
    include_holdings_context: bool = False,
) -> None:
    if not ticker:
        st.info("Add a ticker on the left to view valuation analysis.")
        return

    if result is None:
        st.info("Click an asset on the left to analyze and display it.")
        return

    st.header(f"Valuation Analysis: {ticker}")
    if result.issue:
        st.error(result.issue.message)
        if result.issue.details:
            st.caption(result.issue.details)
        return

    st.markdown('<div class="pane-copy">Selected asset output appears here after you click a ticker in the left pane.</div>', unsafe_allow_html=True)
    left, right = st.columns([0.9, 2.8], vertical_alignment="top")
    with left:
        if result.fundamentals is not None:
            fundamentals_df = fundamentals_to_frame(
                [result.fundamentals],
                include_eps=True,
                include_holdings=include_holdings_context,
                include_return_windows=include_holdings_context,
            )
            display_df = fundamentals_table_display(fundamentals_df)
            st.dataframe(
                display_df,
                width="stretch",
                height=dataframe_height(len(display_df)),
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    ticker: st.column_config.TextColumn(ticker, width="small"),
                },
            )
            render_export_controls(
                label_prefix=f"{ticker}-valuation-fundamentals",
                csv_name=f"{ticker.lower()}_valuation_fundamentals.csv",
                csv_data=dataframe_to_csv_bytes(fundamentals_df),
            )
    with right:
        st.subheader("Valuation Overview")
        if result.pe_figure is not None:
            st.plotly_chart(result.pe_figure, width="stretch")
            render_export_controls(
                label_prefix=f"{ticker}-valuation-chart",
                html_name=f"{ticker.lower()}_valuation_chart.html",
                html_data=figure_to_html(result.pe_figure),
            )
        elif result.pe_issue is not None:
            st.warning(result.pe_issue.message)
        if result.ps_issue is not None:
            st.caption(f"Revenue bar series unavailable: {result.ps_issue.message}")
        if result.fcf_issue is not None:
            st.caption(f"Free-cash-flow bar series unavailable: {result.fcf_issue.message}")


def render_summary_table(mode: str, summary_df) -> None:
    selection = st.dataframe(
        summary_df,
        width="stretch",
        height=dataframe_height(len(summary_df), visible_rows=20),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"{mode}-summary-table",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Px": st.column_config.NumberColumn("Px", format="%.2f", width="small"),
            "Yld %": st.column_config.NumberColumn("Yld %", format="%.2f", width="small"),
            "P/E": st.column_config.NumberColumn("P/E", format="%.2f", width="small"),
            "Typ": st.column_config.TextColumn("Typ", width="small"),
            "Ret %": st.column_config.NumberColumn("Ret %", format="%.2f", width="small"),
            "Ret Adj %": st.column_config.NumberColumn("Ret Adj %", format="%.2f", width="small"),
            "Vol %": st.column_config.NumberColumn("Vol %", format="%.2f", width="small"),
            "Alpha %": st.column_config.NumberColumn("Alpha %", format="%.2f", width="small"),
            "Beta": st.column_config.NumberColumn("Beta", format="%.2f", width="small"),
            "Shp": st.column_config.NumberColumn("Shp", format="%.2f", width="small"),
            "Shp Adj": st.column_config.NumberColumn("Shp Adj", format="%.2f", width="small"),
        },
    )

    selected_rows = []
    if selection is not None:
        if hasattr(selection, "selection") and hasattr(selection.selection, "rows"):
            selected_rows = list(selection.selection.rows)
        elif isinstance(selection, dict):
            selected_rows = list(selection.get("selection", {}).get("rows", []))

    if selected_rows:
        row_index = selected_rows[0]
        ticker = str(summary_df.iloc[row_index]["Ticker"])
        st.session_state[f"{mode}_selected_ticker"] = ticker
        st.session_state[f"{mode}_show_summary"] = False
        analyze_single_ticker(mode, ticker)
        st.rerun()


def render_analysis_summary(mode: str) -> None:
    tickers = getattr(current_config(), f"{mode}_analysis").tickers
    results = st.session_state[f"{mode}_results"]
    fundamentals_list = [
        results[ticker].fundamentals
        for ticker in tickers
        if ticker in results and results[ticker].fundamentals is not None
    ]
    if not fundamentals_list:
        st.info("No summary data is available yet for this group.")
        return

    mode_label = "Dividend" if mode == "dividend" else "Valuation"
    st.header(f"{mode_label} Summary")
    st.markdown(
        '<div class="pane-copy">Summary metrics for all analyzed tickers in this group.</div>',
        unsafe_allow_html=True,
    )
    summary_df = build_analysis_summary_frame(fundamentals_list)
    render_summary_table(mode, summary_df)


def render_holdings_summary() -> None:
    st.header("Holdings Analysis")
    st.markdown(
        '<div class="pane-copy">Imported core holdings data from the Investments Google Sheet, enriched with live yfinance fields and calculated portfolio metrics.</div>',
        unsafe_allow_html=True,
    )
    holdings_df = build_holdings_analysis_table()
    totals = compute_holdings_totals(holdings_df)
    holdings_display_df = format_portfolio_holdings_for_display(holdings_df)
    holdings_selection = st.dataframe(
        holdings_display_df,
        width="stretch",
        height=dataframe_height(len(holdings_display_df), visible_rows=20),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="holdings-summary-table",
    )
    selected_rows = []
    if holdings_selection is not None:
        if hasattr(holdings_selection, "selection") and hasattr(holdings_selection.selection, "rows"):
            selected_rows = list(holdings_selection.selection.rows)
        elif isinstance(holdings_selection, dict):
            selected_rows = list(holdings_selection.get("selection", {}).get("rows", []))
    if selected_rows:
        row_index = selected_rows[0]
        source_row = holdings_df.iloc[row_index]
        ticker = str(source_row.get("Ticker", "")).strip().upper()
        asset_type = source_row.get("Type")
        if holdings_analysis_mode(asset_type) is not None and ticker:
            switch_holdings_detail(ticker, asset_type)
            st.rerun()
    st.markdown(
        f'''
        <div class="totals-row">
            <span><strong>Total Market Value:</strong> ${totals["market_value"]:,.2f}</span>
            <span><strong>Total Income:</strong> ${totals["income"]:,.2f}</span>
            <span><strong>Total Gain:</strong> {totals["total_gain"] * 100:.2f}%</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    income_by_month_df = build_holdings_income_by_month_table()
    if not income_by_month_df.empty:
        st.divider()
        st.subheader("Income by Month")
        income_display_df = format_income_by_month_for_display(income_by_month_df).drop(
            columns=["Ticker", "Type"],
            errors="ignore",
        )
        income_selection = st.dataframe(
            income_display_df,
            width="stretch",
            height=dataframe_height(len(income_display_df), visible_rows=20),
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="holdings-income-by-month-table",
        )
        selected_rows = []
        if income_selection is not None:
            if hasattr(income_selection, "selection") and hasattr(income_selection.selection, "rows"):
                selected_rows = list(income_selection.selection.rows)
            elif isinstance(income_selection, dict):
                selected_rows = list(income_selection.get("selection", {}).get("rows", []))
        if selected_rows:
            row_index = selected_rows[0]
            source_row = income_by_month_df.iloc[row_index]
            ticker = str(source_row.get("Ticker", "")).strip().upper()
            asset_type = source_row.get("Type")
            if holdings_analysis_mode(asset_type) is not None and ticker:
                switch_holdings_detail(ticker, asset_type)
                st.rerun()
    window_start = analysis_start_date()
    window_end = pd.Timestamp(date.today()).date()
    portfolio_value_history, monthly_income_history = build_holdings_portfolio_window(
        window_start,
        window_end,
        assume_full_period=False,
    )
    if not portfolio_value_history.empty:
        st.divider()
        st.subheader("Portfolio Performance")
        st.plotly_chart(
            build_holdings_portfolio_chart(
                portfolio_value_history,
                monthly_income_history,
                title="Holdings Portfolio - Current Holdings Since Recorded Buy Dates (Estimated)",
                show_income_bars=not lookback_exceeds_years(current_lookback(), 2),
            ),
            width="stretch",
        )
    full_year_value_history, full_year_income_history = build_holdings_portfolio_window(
        window_start,
        window_end,
        assume_full_period=True,
    )
    if not full_year_value_history.empty:
        spy_benchmark_series = build_rebased_benchmark_series(BENCHMARK_TICKER, full_year_value_history)
        st.subheader("Portfolio Performance Assuming Current Holdings Were Held All Period")
        st.plotly_chart(
            build_holdings_portfolio_chart(
                full_year_value_history,
                full_year_income_history,
                title="Holdings Portfolio - Current Holdings Held for Full Period",
                show_income_bars=not lookback_exceeds_years(current_lookback(), 2),
                benchmark_history=spy_benchmark_series,
            ),
            width="stretch",
        )
        metrics_df = build_holdings_window_metrics_table(
            build_holdings_portfolio_histories()["full_year"][0],
            build_holdings_portfolio_histories()["full_year"][1],
        )
        if not metrics_df.empty:
            metrics_display_df = format_holdings_window_metrics_table(metrics_df)
            metrics_column_config = {
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
            }
            for column in metrics_display_df.columns:
                if column == "Metric":
                    continue
                metrics_column_config[column] = st.column_config.TextColumn(column, width="small")
            st.subheader("Portfolio Metrics by Window")
            st.dataframe(
                metrics_display_df,
                width="stretch",
                height=dataframe_height(len(metrics_df.reset_index()), visible_rows=10),
                hide_index=True,
                column_config=metrics_column_config,
            )


def render_portfolio_tab() -> None:
    if not st.session_state.get("holdings_refreshed_on_load", False):
        refresh_holdings_analysis_data()
        st.session_state["holdings_refreshed_on_load"] = True

    ensure_holdings_selection()
    sync_holdings_detail()

    left, right = st.columns([0.9, 3.4], vertical_alignment="top")
    with left:
        portfolio_pane = st.container(border=True)
        with portfolio_pane:
            render_holdings_asset_list()
    with right:
        holdings_pane = st.container(border=True)
        with holdings_pane:
            try:
                if st.session_state.get("holdings_show_summary", True):
                    render_holdings_summary()
                else:
                    ticker = st.session_state.get("holdings_selected_ticker")
                    asset_type = st.session_state.get("holdings_selected_type")
                    mode = holdings_analysis_mode(asset_type)
                    if mode == "dividend":
                        result = st.session_state["dividend_results"].get(ticker) if ticker else None
                        render_dividend_main(result, ticker, include_holdings_context=True)
                    elif mode == "valuation":
                        result = st.session_state["valuation_results"].get(ticker) if ticker else None
                        render_valuation_main(result, ticker, include_holdings_context=True)
                    else:
                        render_holdings_summary()
            except Exception as exc:
                st.error(str(exc))


def render_analysis_tab(mode: str) -> None:
    ensure_ticker_selection(mode)
    settings = getattr(current_config(), f"{mode}_analysis")
    prune_results_for_mode(mode, settings.tickers)
    sync_initial_analyses(mode)

    left, right = st.columns([0.9, 3.4], vertical_alignment="top")
    with left:
        asset_pane = st.container(border=True)
        with asset_pane:
            render_asset_list(mode)
    with right:
        detail_pane = st.container(border=True)
        selected_ticker = st.session_state.get(f"{mode}_selected_ticker")
        result = st.session_state[f"{mode}_results"].get(selected_ticker) if selected_ticker else None
        with detail_pane:
            if st.session_state.get(f"{mode}_show_summary", False):
                render_analysis_summary(mode)
            elif mode == "dividend":
                render_dividend_main(result, selected_ticker)
            else:
                render_valuation_main(result, selected_ticker)


def render_page() -> None:
    inject_page_styles()
    render_page_header()
    warm_market_data_on_load()
    st.caption("Ticker lists are saved in `config.json`. Historical price data is cached in `.cache/`.")
    st.radio(
        "Lookback",
        options=LOOKBACK_OPTIONS,
        key="selected_lookback",
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_page = st.segmented_control(
        "Analysis Page",
        options=["Dividend Analysis", "Valuation Analysis", "Holdings Analysis"],
        key="selected_page",
        label_visibility="collapsed",
    )
    if selected_page == "Dividend Analysis":
        render_analysis_tab("dividend")
    elif selected_page == "Valuation Analysis":
        render_analysis_tab("valuation")
    else:
        render_portfolio_tab()


initialize_state()
render_page()
