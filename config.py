from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from models import AnalysisSettings, AppConfig


CONFIG_FILE = "config.json"


def default_config() -> AppConfig:
    return AppConfig(
        dividend_analysis=AnalysisSettings(
            tickers=["VOO", "BKLN", "JEPI"],
            start_date=date.today() - timedelta(days=365),
        ),
        valuation_analysis=AnalysisSettings(
            tickers=["MSFT", "AAPL"],
            start_date=date.today() - timedelta(days=365 * 5),
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

    return AppConfig(
        dividend_analysis=AnalysisSettings(
            tickers=_parse_tickers(dividend_raw.get("tickers"), defaults.dividend_analysis.tickers),
            start_date=_parse_date(
                dividend_raw.get("start_date"),
                defaults.dividend_analysis.start_date,
            ),
        ),
        valuation_analysis=AnalysisSettings(
            tickers=_parse_tickers(valuation_raw.get("tickers"), defaults.valuation_analysis.tickers),
            start_date=_parse_date(
                valuation_raw.get("start_date"),
                defaults.valuation_analysis.start_date,
            ),
        ),
    )


def save_config(config: AppConfig, path: str = CONFIG_FILE) -> None:
    payload = {
        "dividend_analysis": {
            "tickers": config.dividend_analysis.tickers,
            "start_date": config.dividend_analysis.start_date.strftime("%Y-%m-%d"),
        },
        "pe_analysis": {
            "tickers": config.valuation_analysis.tickers,
            "start_date": config.valuation_analysis.start_date.strftime("%Y-%m-%d"),
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
