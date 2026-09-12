from datetime import date

import pandas as pd
import pytest

import portfolio_data


def _holdings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": "OLD",
                "Type": "stock",
                "Loc": "Taxable",
                "Quantity": 1,
                "Buy Date": "2020-01-01",
                "Bought at": 5,
            },
            {
                "Ticker": "NEW",
                "Type": "stock",
                "Loc": "Taxable",
                "Quantity": 2,
                "Buy Date": "2020-01-01",
                "Bought at": 50,
            },
        ]
    )


def _history(prices: dict[str, float]) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(list(prices)))
    return pd.DataFrame(
        {
            "Close": list(prices.values()),
            "Dividends": 0.0,
        },
        index=index,
    )


@pytest.fixture
def portfolio_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "OLD": _history(
            {
                "2026-01-05": 10.0,
                "2026-01-06": 11.0,
                "2026-01-07": 12.0,
                "2026-01-08": 13.0,
                "2026-01-09": 14.0,
            }
        ),
        "NEW": _history(
            {
                "2026-01-08": 100.0,
                "2026-01-09": 110.0,
            }
        ),
    }

    monkeypatch.setattr(portfolio_data, "load_portfolio_holdings", _holdings)
    monkeypatch.setattr(portfolio_data, "get_full_price_history", histories.__getitem__)
    monkeypatch.setattr(
        portfolio_data,
        "_session_index",
        lambda start, end: pd.bdate_range(start=start, end=end),
    )


@pytest.mark.parametrize("assume_full_period", [False, True])
def test_portfolio_window_does_not_value_ticker_before_first_price(
    portfolio_inputs: None,
    assume_full_period: bool,
) -> None:
    values, _ = portfolio_data.build_holdings_portfolio_window.__wrapped__(
        date(2026, 1, 5),
        date(2026, 1, 9),
        assume_full_period,
    )

    assert values["Portfolio Value"].tolist() == [10.0, 11.0, 12.0, 213.0, 234.0]
    assert values["Reinvested Portfolio Value"].tolist() == [10.0, 11.0, 12.0, 213.0, 234.0]


def test_full_history_does_not_value_ticker_before_first_price(
    portfolio_inputs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 1, 9)

    monkeypatch.setattr(portfolio_data, "date", FixedDate)

    values, _ = portfolio_data.build_holdings_portfolio_histories.__wrapped__()["full_year"]

    assert (values.loc[values.index < "2026-01-05"] == 0.0).all().all()
    expected = [10.0, 11.0, 12.0, 213.0, 234.0]
    assert values.loc["2026-01-05":, "Portfolio Value"].tolist() == expected
    assert values.loc["2026-01-05":, "Reinvested Portfolio Value"].tolist() == expected


def test_portfolio_window_includes_sold_position_only_through_sell_date(
    portfolio_inputs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        portfolio_data,
        "load_portfolio_sold",
        lambda: pd.DataFrame(
            [
                {
                    "Ticker": "OLD",
                    "Type": "stock",
                    "Loc": "Taxable",
                    "Quantity": 3,
                    "Buy Date": "2026-01-06",
                    "Bought at": 10,
                    "Sell Date": "2026-01-08",
                }
            ]
        ),
    )

    values, _ = portfolio_data.build_holdings_portfolio_window.__wrapped__(
        date(2026, 1, 5),
        date(2026, 1, 9),
        assume_full_period=False,
        include_sold=True,
    )

    assert values["Portfolio Value"].tolist() == [10.0, 44.0, 48.0, 252.0, 234.0]


def test_cash_history_rolls_current_balance_backward_using_net_cash_flows() -> None:
    holdings = pd.DataFrame(
        [
            {
                "Ticker": "DTMA",
                "Type": "Cash",
                "Loc": "401K",
                "Quantity": 100,
            }
        ]
    )
    cash_flows = pd.DataFrame(
        [
            {"Loc": "401K", "Date": date(2026, 1, 8), "Amount": -50},
            {"Loc": "401K", "Date": date(2026, 1, 9), "Amount": 20},
        ]
    )
    index = pd.bdate_range("2026-01-05", "2026-01-09")

    cash = portfolio_data._reconstruct_cash_history(
        holdings,
        cash_flows,
        index,
        date(2026, 1, 9),
    )

    assert cash.tolist() == [130.0, 130.0, 130.0, 80.0, 100.0]


def test_cash_flow_parser_keeps_trades_but_excludes_sweeps_and_income() -> None:
    values = [
        ["Trade Date", "Transaction Type", "Net Amount"],
        ["2026-02-17", "Buy", "-$50,000.00"],
        ["2026-02-18", "Sweep out", "$50,000.00"],
        ["2026-02-27", "Dividend", "$100.00"],
        ["2026-03-01", "Sell", "$51,000.00"],
    ]

    rows = portfolio_data._cash_flow_rows(
        values,
        "Trade Date",
        "Net Amount",
        "Transaction Type",
    )

    assert rows == [
        {"Date": date(2026, 2, 17), "Amount": -50000.0},
        {"Date": date(2026, 3, 1), "Amount": 51000.0},
    ]
