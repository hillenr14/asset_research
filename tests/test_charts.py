import pandas as pd

from charts import build_holdings_portfolio_chart


def test_holdings_chart_can_hide_reinvestment_and_centers_income_in_month() -> None:
    value_index = pd.bdate_range("2026-05-01", "2026-07-31")
    values = pd.DataFrame(
        {
            "Portfolio Value": 100.0,
            "Reinvested Portfolio Value": 110.0,
        },
        index=value_index,
    )
    income = pd.DataFrame(
        {"Income": [500.0, 600.0]},
        index=pd.to_datetime(["2026-05-31", "2026-06-30"]),
    )

    figure = build_holdings_portfolio_chart(
        values,
        income,
        show_reinvested_value=False,
    )

    assert [trace.name for trace in figure.data] == [
        "Portfolio Value",
        "Estimated Monthly Income",
    ]
    assert list(figure.data[1].x) == [
        pd.Timestamp("2026-05-15"),
        pd.Timestamp("2026-06-15"),
    ]
    assert list(figure.data[1].customdata) == ["May 2026", "June 2026"]
