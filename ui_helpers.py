from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go

from models import DividendAnalysisResult, Fundamentals, ValuationAnalysisResult
from ticker_symbols import ticker_symbol_is_valid


def normalize_single_ticker(text: str) -> str:
    return text.strip().upper()


def parse_tickers(text: str) -> tuple[list[str], list[str]]:
    tickers: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        ticker = line.strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            duplicates.append(ticker)
            continue
        seen.add(ticker)
        tickers.append(ticker)

    return tickers, duplicates


def validate_analysis_request(tickers: list[str], start_date: date) -> list[str]:
    errors: list[str] = []
    if not tickers:
        errors.append("Enter at least one ticker.")
    if start_date > date.today():
        errors.append("Start date cannot be in the future.")
    return errors


def validate_single_ticker(ticker: str) -> list[str]:
    errors: list[str] = []
    if not ticker:
        errors.append("Enter a ticker before adding it to the list.")
    elif " " in ticker:
        errors.append("Enter one ticker at a time.")
    elif not ticker_symbol_is_valid(ticker):
        errors.append("Ticker contains unsupported characters.")
    return errors


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    def _format_value(value):
        if isinstance(value, (int, float)):
            return f"{value:.2f}" if not pd.isna(value) else "N/A"
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "N/A"
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    display_df = df.copy().astype("object")
    for column in display_df.columns:
        display_df[column] = display_df[column].map(_format_value)
    return display_df


def dividend_status_frame(results: list[DividendAnalysisResult]) -> pd.DataFrame:
    records = []
    for result in results:
        records.append(
            {
                "Ticker": result.ticker,
                "Status": "Ready" if result.is_success else "Error",
                "Category": result.issue.category if result.issue else "",
                "Message": result.issue.message if result.issue else "Dividend analysis complete.",
            }
        )
    return pd.DataFrame(records)


def valuation_status_frame(results: list[ValuationAnalysisResult]) -> pd.DataFrame:
    records = []
    for result in results:
        records.append(
            {
                "Ticker": result.ticker,
                "Overall": result.status.title(),
                "P/E": "Ready" if result.pe_figure is not None else (result.pe_issue.category if result.pe_issue else ""),
                "Revenue": "Ready" if result.ps_issue is None else result.ps_issue.category,
                "FCF": "Ready" if result.fcf_issue is None else result.fcf_issue.category,
                "Message": (
                    result.issue.message
                    if result.issue
                    else "; ".join(
                        issue.message for issue in [result.pe_issue, result.ps_issue, result.fcf_issue] if issue is not None
                    )
                    or "Valuation analysis complete."
                ),
            }
        )
    return pd.DataFrame(records)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def figure_to_html(figure: go.Figure) -> str:
    return figure.to_html(include_plotlyjs="cdn")


def build_analysis_summary_frame(fundamentals_list: list[Fundamentals]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fundamentals in fundamentals_list:
        records.append(
            {
                "Ticker": fundamentals.ticker,
                "Name": fundamentals.name,
                "Px": fundamentals.price,
                "Yld %": (fundamentals.dividend_yield * 100) if fundamentals.dividend_yield is not None else None,
                "P/E": fundamentals.trailing_pe,
                "Typ": fundamentals.asset_type,
                "Ret %": fundamentals.annual_return_pct,
                "Ret Adj %": fundamentals.annual_return_adj_pct,
                "Vol %": fundamentals.annual_volatility_pct,
                "Alpha %": fundamentals.alpha_vs_spy_pct,
                "Beta": fundamentals.beta_vs_spy,
                "Shp": fundamentals.sharpe_ratio,
                "Shp Adj": fundamentals.sharpe_ratio_adj,
            }
        )

    return pd.DataFrame(records)
