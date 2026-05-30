from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import plotly.graph_objects as go


@dataclass
class AnalysisSettings:
    tickers: list[str]
    start_date: date


@dataclass
class AppConfig:
    dividend_analysis: AnalysisSettings
    valuation_analysis: AnalysisSettings


@dataclass
class AnalysisIssue:
    category: str
    message: str
    details: Optional[str] = None


@dataclass
class TickerSnapshot:
    ticker: str
    short_name: str
    asset_type: str
    regular_market_price: Optional[float]
    dividend_yield: Optional[float]
    trailing_pe: Optional[float]
    shares_outstanding: Optional[float]


@dataclass
class Fundamentals:
    ticker: str
    name: str
    price: Optional[float]
    dividend_yield: Optional[float]
    trailing_pe: Optional[float]
    trailing_eps: Optional[float]
    asset_type: str
    start_date: Optional[date]
    end_date: Optional[date]
    annual_return_pct: Optional[float] = None
    annual_return_adj_pct: Optional[float] = None
    annual_volatility_pct: Optional[float] = None
    alpha_vs_spy_pct: Optional[float] = None
    beta_vs_spy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sharpe_ratio_adj: Optional[float] = None
    holdings_quantity: Optional[float] = None
    holdings_buy_date: Optional[date] = None
    holdings_bought_at: Optional[float] = None
    holdings_market_value: Optional[float] = None
    holdings_gain: Optional[float] = None
    holdings_income: Optional[float] = None
    return_1d_pct: Optional[float] = None
    return_1w_pct: Optional[float] = None
    return_1m_pct: Optional[float] = None
    return_3m_pct: Optional[float] = None
    return_6m_pct: Optional[float] = None
    return_1y_pct: Optional[float] = None
    return_2y_pct: Optional[float] = None
    return_5y_pct: Optional[float] = None
    return_10y_pct: Optional[float] = None


@dataclass
class DividendAnalysisResult:
    ticker: str
    fundamentals: Optional[Fundamentals] = None
    history: Optional[pd.DataFrame] = None
    figure: Optional[go.Figure] = None
    issue: Optional[AnalysisIssue] = None

    @property
    def is_success(self) -> bool:
        return self.issue is None and self.fundamentals is not None and self.figure is not None

    @property
    def status(self) -> str:
        return "success" if self.is_success else "error"


@dataclass
class ValuationAnalysisResult:
    ticker: str
    fundamentals: Optional[Fundamentals] = None
    pe_figure: Optional[go.Figure] = None
    ps_figure: Optional[go.Figure] = None
    issue: Optional[AnalysisIssue] = None
    pe_issue: Optional[AnalysisIssue] = None
    ps_issue: Optional[AnalysisIssue] = None
    fcf_issue: Optional[AnalysisIssue] = None

    @property
    def status(self) -> str:
        if self.issue is not None:
            return "error"
        if self.pe_figure is not None and self.pe_issue is None and self.ps_issue is None and self.fcf_issue is None:
            return "success"
        if self.pe_figure is not None:
            return "partial"
        return "error"
