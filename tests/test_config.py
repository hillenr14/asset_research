from datetime import date

from config import DEFAULT_LOOKBACK, lookback_exceeds_years, lookback_start_date


def test_default_lookback_is_two_years() -> None:
    end_date = date(2026, 8, 9)

    assert lookback_start_date(DEFAULT_LOOKBACK, end_date) == date(2024, 8, 9)


def test_all_lookback_uses_earliest_supported_date() -> None:
    assert lookback_start_date("all", date(2026, 8, 9)) == date(1900, 1, 1)
    assert lookback_exceeds_years("all", 10)


def test_unknown_lookback_falls_back_to_default() -> None:
    end_date = date(2026, 8, 9)

    assert lookback_start_date("unknown", end_date) == lookback_start_date(
        DEFAULT_LOOKBACK,
        end_date,
    )
