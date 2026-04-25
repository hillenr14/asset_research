from __future__ import annotations

from datetime import date

import streamlit as st

from config import load_config, save_config
from metrics import analyze_dividend_ticker, analyze_valuation_ticker, fundamentals_to_frame
from models import AnalysisSettings, DividendAnalysisResult, ValuationAnalysisResult
from ui_helpers import (
    dataframe_to_csv_bytes,
    dividend_status_frame,
    figure_to_html,
    format_dataframe_for_display,
    parse_tickers,
    validate_analysis_request,
    valuation_status_frame,
)


st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("Stock Analysis")


def initialize_state() -> None:
    config = load_config()
    st.session_state.setdefault("dividend_results", [])
    st.session_state.setdefault("valuation_results", [])
    st.session_state.setdefault(
        "valuation_tickers",
        config.valuation_analysis.tickers,
    )


def render_export_controls(
    label_prefix: str,
    csv_name: str | None = None,
    csv_data: bytes | None = None,
    html_name: str | None = None,
    html_data: str | None = None,
) -> None:
    if not st.session_state.get("show_export_actions"):
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


def run_dividend_analysis(tickers: list[str], start_date: date) -> list[DividendAnalysisResult]:
    results: list[DividendAnalysisResult] = []
    progress = st.progress(0.0, text="Preparing dividend analysis...")
    for index, ticker in enumerate(tickers, start=1):
        progress.progress(index / len(tickers), text=f"Analyzing dividends for {ticker} ({index}/{len(tickers)})")
        results.append(analyze_dividend_ticker(ticker, start_date))
    progress.empty()
    return results


def run_valuation_analysis(tickers: list[str], start_date: date) -> list[ValuationAnalysisResult]:
    results: list[ValuationAnalysisResult] = []
    progress = st.progress(0.0, text="Preparing valuation analysis...")
    for index, ticker in enumerate(tickers, start=1):
        progress.progress(index / len(tickers), text=f"Analyzing valuation for {ticker} ({index}/{len(tickers)})")
        results.append(analyze_valuation_ticker(ticker, start_date))
    progress.empty()
    return results


def render_sidebar() -> None:
    config = load_config()

    st.sidebar.header("Controls")
    st.sidebar.checkbox("Show export actions", key="show_export_actions")

    with st.sidebar.expander("Dividend & Price Analysis", expanded=True):
        with st.form("dividend-analysis-form"):
            div_tickers_input = st.text_area(
                "Tickers:",
                value="\n".join(config.dividend_analysis.tickers),
                height=120,
            )
            div_start_date = st.date_input(
                "Start Date:",
                value=config.dividend_analysis.start_date,
            )
            submitted = st.form_submit_button("Analyze Dividends & Price")

        if submitted:
            tickers, duplicates = parse_tickers(div_tickers_input)
            validation_errors = validate_analysis_request(tickers, div_start_date)
            if duplicates:
                st.warning(f"Duplicate tickers removed: {', '.join(duplicates)}")
            if validation_errors:
                for message in validation_errors:
                    st.error(message)
            else:
                config.dividend_analysis = AnalysisSettings(tickers=tickers, start_date=div_start_date)
                save_config(config)
                with st.spinner("Running dividend analysis..."):
                    st.session_state.dividend_results = run_dividend_analysis(tickers, div_start_date)

    with st.sidebar.expander("Historical Valuation Analysis", expanded=True):
        with st.form("valuation-analysis-form"):
            valuation_tickers_input = st.text_area(
                "Tickers:",
                value="\n".join(config.valuation_analysis.tickers),
                height=120,
            )
            valuation_start_date = st.date_input(
                "Start Date:",
                value=config.valuation_analysis.start_date,
                key="valuation-start-date",
            )
            submitted = st.form_submit_button("Analyze Valuation Ratios")

        if submitted:
            tickers, duplicates = parse_tickers(valuation_tickers_input)
            validation_errors = validate_analysis_request(tickers, valuation_start_date)
            if duplicates:
                st.warning(f"Duplicate tickers removed: {', '.join(duplicates)}")
            if validation_errors:
                for message in validation_errors:
                    st.error(message)
            else:
                config.valuation_analysis = AnalysisSettings(tickers=tickers, start_date=valuation_start_date)
                save_config(config)
                st.session_state.valuation_tickers = tickers
                with st.spinner("Running valuation analysis..."):
                    st.session_state.valuation_results = run_valuation_analysis(tickers, valuation_start_date)

    st.sidebar.info("Inputs are saved in `config.json`.", icon="💡")


def render_dividend_summary(results: list[DividendAnalysisResult]) -> None:
    if not results:
        st.info("Click 'Analyze Dividends & Price' to see results.")
        return

    st.header("Dividend Analysis Status")
    st.dataframe(dividend_status_frame(results), hide_index=True, width="stretch")

    successful_results = [result for result in results if result.is_success and result.fundamentals is not None]
    if successful_results:
        st.header("Combined Fundamentals & Metrics")
        combined_df = fundamentals_to_frame([result.fundamentals for result in successful_results if result.fundamentals])
        display_df = format_dataframe_for_display(combined_df)
        st.dataframe(display_df, width="stretch")
        render_export_controls(
            label_prefix="combined-dividend-summary",
            csv_name="dividend_summary.csv",
            csv_data=dataframe_to_csv_bytes(combined_df),
        )

    st.header("Individual Charts")
    for result in results:
        st.subheader(f"Analysis for {result.ticker}")
        if result.issue:
            st.error(result.issue.message)
            if result.issue.details:
                st.caption(result.issue.details)
            continue

        if not result.fundamentals or result.figure is None:
            st.warning("No dividend output was generated for this ticker.")
            continue

        detail_df = format_dataframe_for_display(fundamentals_to_frame([result.fundamentals]))
        col1, col2 = st.columns([0.7, 1.8])
        with col1:
            st.dataframe(detail_df, width="stretch")
        with col2:
            st.plotly_chart(result.figure, width="stretch")

        render_export_controls(
            label_prefix=f"{result.ticker}-dividend",
            csv_name=f"{result.ticker.lower()}_dividend_fundamentals.csv",
            csv_data=dataframe_to_csv_bytes(fundamentals_to_frame([result.fundamentals])),
            html_name=f"{result.ticker.lower()}_dividend_chart.html",
            html_data=figure_to_html(result.figure),
        )


def render_valuation_tab(result: ValuationAnalysisResult | None) -> None:
    if result is None:
        st.info("Click 'Analyze Valuation Ratios' to see results.")
        return

    if result.issue:
        st.error(result.issue.message)
        if result.issue.details:
            st.caption(result.issue.details)
        return

    col1, col2 = st.columns([0.7, 1.8])
    with col1:
        st.subheader("Fundamentals")
        if result.fundamentals is not None:
            fundamentals_df = format_dataframe_for_display(fundamentals_to_frame([result.fundamentals]))
            st.dataframe(fundamentals_df, width="stretch")
            render_export_controls(
                label_prefix=f"{result.ticker}-valuation-fundamentals",
                csv_name=f"{result.ticker.lower()}_valuation_fundamentals.csv",
                csv_data=dataframe_to_csv_bytes(fundamentals_to_frame([result.fundamentals])),
            )
        else:
            st.warning("Could not retrieve fundamentals.")

    with col2:
        st.subheader("Price-to-Earnings (P/E) Ratio")
        if result.pe_figure is not None:
            st.plotly_chart(result.pe_figure, width="stretch")
            render_export_controls(
                label_prefix=f"{result.ticker}-pe",
                html_name=f"{result.ticker.lower()}_pe_chart.html",
                html_data=figure_to_html(result.pe_figure),
            )
        elif result.pe_issue is not None:
            st.warning(result.pe_issue.message)

        st.subheader("Price-to-Sales (P/S) Ratio")
        if result.ps_figure is not None:
            st.plotly_chart(result.ps_figure, width="stretch")
            render_export_controls(
                label_prefix=f"{result.ticker}-ps",
                html_name=f"{result.ticker.lower()}_ps_chart.html",
                html_data=figure_to_html(result.ps_figure),
            )
        elif result.ps_issue is not None:
            st.warning(result.ps_issue.message)


def render_main_area() -> None:
    dividend_results: list[DividendAnalysisResult] = st.session_state.dividend_results
    valuation_results: list[ValuationAnalysisResult] = st.session_state.valuation_results
    valuation_by_ticker = {result.ticker: result for result in valuation_results}
    valuation_tickers = st.session_state.valuation_tickers

    tab_names = ["Summary & Dividends"] + [f"{ticker} Valuation" for ticker in valuation_tickers]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_dividend_summary(dividend_results)
        if valuation_results:
            st.header("Valuation Analysis Status")
            st.dataframe(valuation_status_frame(valuation_results), hide_index=True, width="stretch")
            render_export_controls(
                label_prefix="valuation-status",
                csv_name="valuation_status.csv",
                csv_data=dataframe_to_csv_bytes(valuation_status_frame(valuation_results)),
            )

    for index, ticker in enumerate(valuation_tickers, start=1):
        with tabs[index]:
            st.header(f"Valuation Analysis for {ticker}")
            render_valuation_tab(valuation_by_ticker.get(ticker))


initialize_state()
render_sidebar()
render_main_area()
