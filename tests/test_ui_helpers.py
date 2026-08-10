from ui_helpers import normalize_single_ticker, parse_tickers, validate_single_ticker


def test_parse_tickers_normalizes_and_reports_duplicates() -> None:
    tickers, duplicates = parse_tickers(" aapl\nMSFT\nAAPL\n")

    assert tickers == ["AAPL", "MSFT"]
    assert duplicates == ["AAPL"]


def test_single_ticker_normalization() -> None:
    assert normalize_single_ticker("  spy ") == "SPY"
    assert validate_single_ticker("SPY") == []


def test_single_ticker_validation_rejects_cache_path_characters() -> None:
    assert validate_single_ticker("../../outside") == [
        "Ticker contains unsupported characters."
    ]
