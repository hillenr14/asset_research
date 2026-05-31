from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from models import AnalysisSettings, AppConfig


CONFIG_FILE = "config.json"
ROLLING_LOOKBACK_DAYS = 365 * 2
DEFAULT_LOOKBACK = "2y"
LOOKBACK_OPTIONS = ["1w", "1m", "3m", "6m", "1y", "2y", "5y", "10y", "all"]
LOOKBACK_DAY_MAP = {
    "1w": 7,
    "1m": 30,
    "3m": 91,
    "6m": 182,
    "1y": 365,
    "2y": 365 * 2,
    "5y": 365 * 5,
    "10y": 365 * 10,
    "all": None,
}


def rolling_analysis_start_date() -> date:
    return date.today() - timedelta(days=ROLLING_LOOKBACK_DAYS)


def lookback_start_date(lookback: str, end_date: date | None = None) -> date:
    effective_end_date = end_date or date.today()
    days = LOOKBACK_DAY_MAP.get(lookback, LOOKBACK_DAY_MAP[DEFAULT_LOOKBACK])
    if days is None:
        return date(1900, 1, 1)
    return effective_end_date - timedelta(days=days)


def lookback_exceeds_years(lookback: str, years: int) -> bool:
    days = LOOKBACK_DAY_MAP.get(lookback, LOOKBACK_DAY_MAP[DEFAULT_LOOKBACK])
    if days is None:
        return True
    return days > years * 365


def default_config() -> AppConfig:
    return AppConfig(
        dividend_analysis=AnalysisSettings(
            tickers=["VOO", "BKLN", "JEPI"],
            start_date=rolling_analysis_start_date(),
        ),
        valuation_analysis=AnalysisSettings(
            tickers=["MSFT", "AAPL"],
            start_date=rolling_analysis_start_date(),
        ),
    )


def _parse_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return fallback


def _parse_tickers(value: object, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        parsed = [str(item).strip().upper() for item in value if str(item).strip()]
        return parsed or fallback
    return fallback


def load_config(path: str = CONFIG_FILE) -> AppConfig:
    defaults = default_config()
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults

    dividend_raw = raw.get("dividend_analysis", {})
    valuation_raw = raw.get("pe_analysis", raw.get("valuation_analysis", {}))
    rolling_start_date = rolling_analysis_start_date()

    return AppConfig(
        dividend_analysis=AnalysisSettings(
            tickers=_parse_tickers(dividend_raw.get("tickers"), defaults.dividend_analysis.tickers),
            start_date=rolling_start_date,
        ),
        valuation_analysis=AnalysisSettings(
            tickers=_parse_tickers(valuation_raw.get("tickers"), defaults.valuation_analysis.tickers),
            start_date=rolling_start_date,
        ),
    )


def save_config(config: AppConfig, path: str = CONFIG_FILE) -> None:
    rolling_start_date = rolling_analysis_start_date()
    payload = {
        "dividend_analysis": {
            "tickers": config.dividend_analysis.tickers,
            "start_date": rolling_start_date.strftime("%Y-%m-%d"),
        },
        "pe_analysis": {
            "tickers": config.valuation_analysis.tickers,
            "start_date": rolling_start_date.strftime("%Y-%m-%d"),
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
