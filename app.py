from __future__ import annotations

from typing import Callable

import streamlit as st

from charts import build_holdings_portfolio_chart
from config import load_config, rolling_analysis_start_date, save_config
from data_provider import get_ticker_snapshot
from portfolio_data import (
    PORTFOLIO_DOCUMENT_PATH,
    PORTFOLIO_SHEET_NAME,
    PORTFOLIO_TABLE_NAME,
    build_holdings_analysis_table,
    build_holdings_portfolio_histories,
    compute_holdings_totals,
    format_portfolio_holdings_for_display,
    refresh_holdings_analysis_data,
)
from metrics import analyze_dividend_ticker, analyze_valuation_ticker, fundamentals_to_frame
from models import AnalysisSettings, AppConfig, DividendAnalysisResult, ValuationAnalysisResult
from ui_helpers import (
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
        '<div class="app-subtitle">Analyze dividend and valuation signals for individual assets with a two-year rolling history window and persistent local price caching.</div>',
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
    st.session_state["show_export_actions"] = False
    st.session_state.setdefault("dividend_auto_signature", None)
    st.session_state.setdefault("valuation_auto_signature", None)
    st.session_state.setdefault("holdings_refreshed_on_load", False)


def current_config() -> AppConfig:
    return st.session_state.app_config


def persist_current_config() -> None:
    save_config(st.session_state.app_config)


def analysis_start_date():
    return rolling_analysis_start_date()


def get_asset_label(ticker: str) -> str:
    try:
        snapshot, _ = get_ticker_snapshot(ticker)
        if snapshot.short_name and snapshot.short_name != ticker:
            return f"{ticker} · {snapshot.short_name}"
    except Exception:
        pass
    return ticker


def get_asset_name(ticker: str) -> str:
    try:
        snapshot, _ = get_ticker_snapshot(ticker)
        if snapshot.short_name and snapshot.short_name != ticker:
            return snapshot.short_name
    except Exception:
        pass
    return ""


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
    analyze_single_ticker(mode, ticker)


def remove_ticker(mode: str, ticker: str) -> None:
    config = current_config()
    settings = getattr(config, f"{mode}_analysis")
    settings.tickers = [value for value in settings.tickers if value != ticker]
    settings.start_date = analysis_start_date()
    persist_current_config()

    result_key = f"{mode}_results"
    st.session_state[result_key].pop(ticker, None)
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
    signature = tuple(tickers)
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
    st.caption(f"Historical window: {settings.start_date.isoformat()} to today")

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
                analyze_single_ticker(mode, ticker)
                st.rerun()
        with row[2]:
            asset_name = get_asset_name(ticker)
            if asset_name:
                st.markdown(f'<div class="asset-name">{asset_name}</div>', unsafe_allow_html=True)


def render_dividend_main(result: DividendAnalysisResult | None, ticker: str | None) -> None:
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
    detail_df = fundamentals_to_frame([result.fundamentals])
    left, right = st.columns([1.05, 2.65], vertical_alignment="top")
    with left:
        st.dataframe(format_dataframe_for_display(detail_df), width="stretch")
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


def render_valuation_main(result: ValuationAnalysisResult | None, ticker: str | None) -> None:
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
    left, right = st.columns([1.05, 2.65], vertical_alignment="top")
    with left:
        if result.fundamentals is not None:
            fundamentals_df = fundamentals_to_frame([result.fundamentals])
            st.dataframe(format_dataframe_for_display(fundamentals_df), width="stretch")
            render_export_controls(
                label_prefix=f"{ticker}-valuation-fundamentals",
                csv_name=f"{ticker.lower()}_valuation_fundamentals.csv",
                csv_data=dataframe_to_csv_bytes(fundamentals_df),
            )
    with right:
        st.subheader("Price-to-Earnings (P/E) Ratio")
        if result.pe_figure is not None:
            st.plotly_chart(result.pe_figure, width="stretch")
            render_export_controls(
                label_prefix=f"{ticker}-pe",
                html_name=f"{ticker.lower()}_pe_chart.html",
                html_data=figure_to_html(result.pe_figure),
            )
        elif result.pe_issue is not None:
            st.warning(result.pe_issue.message)

        st.subheader("Price-to-Sales (P/S) Ratio")
        if result.ps_figure is not None:
            st.plotly_chart(result.ps_figure, width="stretch")
            render_export_controls(
                label_prefix=f"{ticker}-ps",
                html_name=f"{ticker.lower()}_ps_chart.html",
                html_data=figure_to_html(result.ps_figure),
            )
        elif result.ps_issue is not None:
            st.warning(result.ps_issue.message)


def render_portfolio_tab() -> None:
    if not st.session_state.get("holdings_refreshed_on_load", False):
        refresh_holdings_analysis_data()
        st.session_state["holdings_refreshed_on_load"] = True

    left, right = st.columns([0.9, 3.4], vertical_alignment="top")
    with left:
        portfolio_pane = st.container(border=True)
        with portfolio_pane:
            st.markdown('<div class="pane-title">Portfolio Source</div>', unsafe_allow_html=True)
            st.caption(f"Document: {PORTFOLIO_DOCUMENT_PATH.name}")
            st.caption(f"Sheet: {PORTFOLIO_SHEET_NAME}")
            st.caption(f"Table: {PORTFOLIO_TABLE_NAME}")
    with right:
        holdings_pane = st.container(border=True)
        with holdings_pane:
            st.header("Holdings Analysis")
            st.markdown(
                '<div class="pane-copy">Imported core holdings data from the Numbers workbook, enriched with live yfinance fields and calculated portfolio metrics.</div>',
                unsafe_allow_html=True,
            )
            try:
                holdings_df = build_holdings_analysis_table()
                totals = compute_holdings_totals(holdings_df)
                st.dataframe(
                    format_portfolio_holdings_for_display(holdings_df),
                    width="stretch",
                    hide_index=True,
                )
                st.markdown(
                    f'''
                    <div class="totals-row">
                        <span><strong>Total Market Value:</strong> ${totals["market_value"]:,.2f}</span>
                        <span><strong>Total Income:</strong> ${totals["income"]:,.2f}</span>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
                portfolio_histories = build_holdings_portfolio_histories()
                portfolio_value_history, monthly_income_history = portfolio_histories["actual"]
                if not portfolio_value_history.empty:
                    st.divider()
                    st.subheader("One-Year Portfolio Performance")
                    st.plotly_chart(
                        build_holdings_portfolio_chart(
                            portfolio_value_history,
                            monthly_income_history,
                            title="Holdings Portfolio - Actual Held Period",
                        ),
                        width="stretch",
                    )
                full_year_value_history, full_year_income_history = portfolio_histories["full_year"]
                if not full_year_value_history.empty:
                    st.subheader("One-Year Portfolio Performance Assuming Current Holdings Were Held All Year")
                    st.plotly_chart(
                        build_holdings_portfolio_chart(
                            full_year_value_history,
                            full_year_income_history,
                            title="Holdings Portfolio - Current Holdings Held for Full Year",
                        ),
                        width="stretch",
                    )
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
            if mode == "dividend":
                render_dividend_main(result, selected_ticker)
            else:
                render_valuation_main(result, selected_ticker)


def render_page() -> None:
    inject_page_styles()
    render_page_header()
    st.caption("Ticker lists are saved in `config.json`. Historical price data is cached in `.cache/`.")

    dividend_tab, valuation_tab, portfolio_tab = st.tabs(
        ["Dividend Analysis", "Valuation Analysis", "Holdings Analysis"]
    )

    with dividend_tab:
        render_analysis_tab("dividend")
    with valuation_tab:
        render_analysis_tab("valuation")
    with portfolio_tab:
        render_portfolio_tab()


initialize_state()
render_page()
