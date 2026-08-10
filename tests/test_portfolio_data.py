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
