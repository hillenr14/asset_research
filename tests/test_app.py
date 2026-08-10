from unittest.mock import Mock

import pandas as pd
from streamlit.testing.v1 import AppTest

import metrics
import portfolio_data
from models import AnalysisIssue, DividendAnalysisResult


def test_initial_dividend_page_is_lazy_and_independent_of_google_sheets(
    monkeypatch,
) -> None:
    def fail_if_holdings_are_loaded() -> pd.DataFrame:
        raise AssertionError("Dividend startup must not load Google holdings")

    analyzer = Mock(
        side_effect=lambda ticker, _start_date, **_kwargs: DividendAnalysisResult(
            ticker=ticker,
            issue=AnalysisIssue("missing_data", "fixture result"),
        )
    )
    monkeypatch.setattr(portfolio_data, "load_portfolio_holdings", fail_if_holdings_are_loaded)
    monkeypatch.setattr(metrics, "analyze_dividend_ticker", analyzer)

    app = AppTest.from_file("app.py", default_timeout=15).run()

    assert len(app.exception) == 0
    assert analyzer.call_count == 1
    assert analyzer.call_args.kwargs == {"include_holdings_detail": False}
