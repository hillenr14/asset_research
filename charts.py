from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import TickerSnapshot

TARGET_BAR_WIDTH_PX = 6
ESTIMATED_PLOT_WIDTH_PX = 950
VALUATION_TARGET_BAR_WIDTH_PX = 16
VALUATION_TARGET_BAR_CENTER_SPACING_PX = 18


def _bar_width_ms(reference_index: pd.Index) -> float:
    if len(reference_index) < 2:
        return 6 * 24 * 60 * 60 * 1000
    start = pd.Timestamp(reference_index.min())
    end = pd.Timestamp(reference_index.max())
    span_ms = max((end - start).total_seconds() * 1000, 24 * 60 * 60 * 1000)
    return max(span_ms * TARGET_BAR_WIDTH_PX / ESTIMATED_PLOT_WIDTH_PX, 1.0)


def _px_to_ms(reference_index: pd.Index, pixels: float) -> float:
    if len(reference_index) < 2:
        return max(pixels, 1.0) * 24 * 60 * 60 * 1000
    start = pd.Timestamp(reference_index.min())
    end = pd.Timestamp(reference_index.max())
    span_ms = max((end - start).total_seconds() * 1000, 24 * 60 * 60 * 1000)
    return max(span_ms * pixels / ESTIMATED_PLOT_WIDTH_PX, 1.0)


def _offset_datetimes_ms(index: pd.Index, offset_ms: float) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(index) + pd.to_timedelta(offset_ms, unit="ms"))


def _apply_plotly_layout(
    fig: go.Figure,
    title: str,
    left_title: str,
    right_title: str | None = None,
    extra_axis: dict | None = None,
) -> go.Figure:
    layout = dict(
        title=title,
        template="plotly",
        hovermode="x unified",
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#142033",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=70, t=90, b=60),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.16)",
            hoverformat="%Y-%m-%d",
            zeroline=False,
        ),
        yaxis=dict(
            title=left_title,
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.16)",
            zeroline=False,
        ),
    )
    if right_title:
        layout["yaxis2"] = dict(
            title=right_title,
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        )
    fig.update_layout(**layout)
    if extra_axis:
        fig.update_layout(yaxis3=extra_axis)
    return fig


def build_ps_chart(
    snapshot: TickerSnapshot,
    price_plot: pd.Series,
    ps_plot: pd.Series,
    revenue_plot: pd.Series,
    show_revenue_bars: bool = True,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_width_ms = _bar_width_ms(price_plot.index)
    fig.add_trace(
        go.Scatter(
            x=ps_plot.index,
            y=ps_plot.values,
            name="Trailing P/S",
            mode="lines",
            line=dict(color="#7dd3fc", width=2),
            hovertemplate="Trailing P/S=%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=price_plot.index,
            y=price_plot.values,
            name="Price",
            mode="lines",
            line=dict(color="#fbbf24", width=2),
            hovertemplate="Price=$%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    if show_revenue_bars:
        fig.add_trace(
            go.Bar(
                x=revenue_plot.index,
                y=revenue_plot.values,
                name="Quarterly Revenue",
                yaxis="y3",
                marker=dict(color="rgba(167, 139, 250, 0.55)"),
                width=bar_width_ms,
                hovertemplate="Revenue=%{y:,.0f}<extra></extra>",
            )
        )
    return _apply_plotly_layout(
        fig,
        f"{snapshot.short_name} ({snapshot.ticker}) - Price, Trailing P/S, and Quarterly Revenue",
        "P/S Ratio",
        "Price",
        extra_axis=dict(
            title="Quarterly Revenue",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.97,
            showgrid=False,
            zeroline=False,
        ),
    )


def build_pe_chart(
    snapshot: TickerSnapshot,
    price_plot: pd.Series,
    pe_plot: pd.Series,
    eps_plot: pd.Series,
    show_eps_bars: bool = True,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_width_ms = _bar_width_ms(price_plot.index)
    fig.add_trace(
        go.Scatter(
            x=pe_plot.index,
            y=pe_plot.values,
            name="Trailing P/E",
            mode="lines",
            line=dict(color="#7dd3fc", width=2),
            hovertemplate="Trailing P/E=%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=price_plot.index,
            y=price_plot.values,
            name="Price",
            mode="lines",
            line=dict(color="#fbbf24", width=2),
            hovertemplate="Price=$%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    if show_eps_bars:
        fig.add_trace(
            go.Bar(
                x=eps_plot.index,
                y=eps_plot.values,
                name="Quarterly EPS",
                yaxis="y3",
                marker=dict(color="rgba(52, 211, 153, 0.55)"),
                width=bar_width_ms,
                hovertemplate="EPS=%{y:.2f}<extra></extra>",
            )
        )
    return _apply_plotly_layout(
        fig,
        f"{snapshot.short_name} ({snapshot.ticker}) - Price, Trailing P/E, and Quarterly EPS",
        "P/E Ratio",
        "Price",
        extra_axis=dict(
            title="Quarterly EPS",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.97,
            showgrid=False,
            zeroline=False,
        ),
    )


def build_valuation_chart(
    snapshot: TickerSnapshot,
    price_plot: pd.Series,
    pe_plot: pd.Series,
    eps_pct_plot: pd.Series | None = None,
    revenue_pct_plot: pd.Series | None = None,
    free_cash_flow_pct_plot: pd.Series | None = None,
    show_quarterly_bars: bool = True,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_width_ms = _px_to_ms(price_plot.index, VALUATION_TARGET_BAR_WIDTH_PX)
    bar_center_spacing_ms = _px_to_ms(price_plot.index, VALUATION_TARGET_BAR_CENTER_SPACING_PX)
    fig.add_trace(
        go.Scatter(
            x=pe_plot.index,
            y=pe_plot.values,
            name="Trailing P/E",
            mode="lines",
            line=dict(color="#7dd3fc", width=2),
            hovertemplate="Trailing P/E=%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=price_plot.index,
            y=price_plot.values,
            name="Price",
            mode="lines",
            line=dict(color="#fbbf24", width=2),
            hovertemplate="Price=$%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    if show_quarterly_bars and eps_pct_plot is not None and not eps_pct_plot.empty:
        fig.add_trace(
            go.Bar(
                x=_offset_datetimes_ms(eps_pct_plot.index, -bar_center_spacing_ms),
                y=eps_pct_plot.values,
                name="EPS / Price",
                yaxis="y3",
                marker=dict(color="rgba(52, 211, 153, 0.55)"),
                text=[f"{value:.2f}%" for value in eps_pct_plot.values],
                textposition="outside",
                textfont=dict(size=11),
                constraintext="none",
                cliponaxis=False,
                width=bar_width_ms,
                hovertemplate="EPS / Price=%{y:.2f}%<extra></extra>",
            )
        )
    if show_quarterly_bars and revenue_pct_plot is not None and not revenue_pct_plot.empty:
        fig.add_trace(
            go.Bar(
                x=_offset_datetimes_ms(revenue_pct_plot.index, 0),
                y=revenue_pct_plot.values,
                name="Rev/Share / Price",
                yaxis="y3",
                marker=dict(color="rgba(167, 139, 250, 0.55)"),
                text=[f"{value:.2f}%" for value in revenue_pct_plot.values],
                textposition="outside",
                textfont=dict(size=11),
                constraintext="none",
                cliponaxis=False,
                width=bar_width_ms,
                hovertemplate="Revenue / Price=%{y:.2f}%<extra></extra>",
            )
        )
    if show_quarterly_bars and free_cash_flow_pct_plot is not None and not free_cash_flow_pct_plot.empty:
        fig.add_trace(
            go.Bar(
                x=_offset_datetimes_ms(free_cash_flow_pct_plot.index, bar_center_spacing_ms),
                y=free_cash_flow_pct_plot.values,
                name="FCF / Price",
                yaxis="y3",
                marker=dict(color="rgba(244, 114, 182, 0.60)"),
                text=[f"{value:.2f}%" for value in free_cash_flow_pct_plot.values],
                textposition="outside",
                textfont=dict(size=11),
                constraintext="none",
                cliponaxis=False,
                width=bar_width_ms,
                hovertemplate="FCF / Price=%{y:.2f}%<extra></extra>",
            )
        )
    fig = _apply_plotly_layout(
        fig,
        f"{snapshot.short_name} ({snapshot.ticker}) - Price, Trailing P/E, and Quarterly Metrics",
        "P/E Ratio",
        "Price",
        extra_axis=dict(
            title="Quarterly % of Price",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.97,
            showgrid=False,
            zeroline=False,
            ticksuffix="%",
        ),
    )
    fig.update_layout(barmode="overlay", bargap=0, bargroupgap=0)
    return fig


def build_dividend_chart(
    snapshot: TickerSnapshot,
    price_history: pd.DataFrame,
    dividends_to_plot: pd.DataFrame,
    bar_labels: list[str],
    benchmark_series: pd.Series | None = None,
    show_dividend_bars: bool = True,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_width_ms = _bar_width_ms(price_history.index)
    fig.add_trace(
        go.Scatter(
            x=price_history.index,
            y=price_history["Close"],
            name="Close",
            mode="lines",
            line=dict(color="#7dd3fc", width=2),
            hovertemplate="Close=$%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=price_history.index,
            y=price_history["Adj Close Rebased"],
            name="Adj Close (rebased)",
            mode="lines",
            line=dict(color="#fbbf24", width=2),
            hovertemplate="Adj Close=$%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    if benchmark_series is not None and not benchmark_series.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_series.index,
                y=benchmark_series.values,
                name="SPY Adj Close (rebased)",
                mode="lines",
                line=dict(color="#ef4444", width=2),
                hovertemplate="SPY Rebased=$%{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )
    if show_dividend_bars:
        fig.add_trace(
            go.Bar(
                x=dividends_to_plot.index,
                y=dividends_to_plot["Dividends"],
                name="Dividends",
                marker=dict(color="rgba(52, 211, 153, 0.55)"),
                text=bar_labels if bar_labels else None,
                textposition="outside",
                textfont=dict(size=12),
                constraintext="none",
                cliponaxis=False,
                width=bar_width_ms,
                hovertemplate="Dividend=$%{y:.4f}<extra></extra>",
            ),
            secondary_y=True,
        )
    fig = _apply_plotly_layout(
        fig,
        f"{snapshot.short_name} ({snapshot.ticker}) - Adjusted Close and Dividends",
        "Close Price",
        "Dividend Amount ($)",
    )
    fig.update_yaxes(rangemode="tozero", secondary_y=True)
    return fig


def build_holdings_portfolio_chart(
    portfolio_value_history: pd.DataFrame,
    monthly_income_history: pd.DataFrame,
    title: str = "Holdings Portfolio - Total Value and Monthly Income",
    show_income_bars: bool = True,
    benchmark_history: pd.Series | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_width_ms = _bar_width_ms(portfolio_value_history.index)
    fig.add_trace(
        go.Scatter(
            x=portfolio_value_history.index,
            y=portfolio_value_history["Portfolio Value"],
            name="Portfolio Value",
            mode="lines",
            line=dict(color="#7dd3fc", width=2),
            hovertemplate="Portfolio Value=$%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    if "Reinvested Portfolio Value" in portfolio_value_history.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_value_history.index,
                y=portfolio_value_history["Reinvested Portfolio Value"],
                name="Value With Reinvestment",
                mode="lines",
                line=dict(color="#f472b6", width=2),
                hovertemplate="Reinvested Value=$%{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )
    if benchmark_history is not None and not benchmark_history.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_history.index,
                y=benchmark_history.values,
                name="SPY Adj Close (rebased)",
                mode="lines",
                line=dict(color="#fbbf24", width=2),
                hovertemplate="SPY Rebased=$%{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

    income_labels = [
        f"${value:,.2f}" if value > 0 else ""
        for value in monthly_income_history["Income"].fillna(0.0)
    ]
    if show_income_bars:
        fig.add_trace(
            go.Bar(
                x=monthly_income_history.index,
                y=monthly_income_history["Income"],
                name="Monthly Income",
                marker=dict(color="rgba(52, 211, 153, 0.55)"),
                text=income_labels,
                textposition="outside",
                textfont=dict(size=12),
                constraintext="none",
                cliponaxis=False,
                width=bar_width_ms,
                hovertemplate="Monthly Income=$%{y:,.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig = _apply_plotly_layout(
        fig,
        title,
        "Portfolio Value ($)",
        "Monthly Income ($)",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, secondary_y=False)
    fig.update_yaxes(
        tickprefix="$",
        separatethousands=True,
        rangemode="tozero",
        secondary_y=True,
    )
    return fig
