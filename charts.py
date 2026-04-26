from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import TickerSnapshot


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
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
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
    fig.add_trace(
        go.Bar(
            x=revenue_plot.index,
            y=revenue_plot.values,
            name="Quarterly Revenue",
            yaxis="y3",
            marker=dict(color="rgba(167, 139, 250, 0.55)"),
            width=10 * 24 * 60 * 60 * 1000,
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
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
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
    fig.add_trace(
        go.Bar(
            x=eps_plot.index,
            y=eps_plot.values,
            name="Quarterly EPS",
            yaxis="y3",
            marker=dict(color="rgba(52, 211, 153, 0.55)"),
            width=10 * 24 * 60 * 60 * 1000,
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


def build_dividend_chart(
    snapshot: TickerSnapshot,
    price_history: pd.DataFrame,
    dividends_to_plot: pd.DataFrame,
    bar_labels: list[str],
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
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
            width=10 * 24 * 60 * 60 * 1000,
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
