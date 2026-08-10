from __future__ import annotations

import re

from errors import InvalidTickerError


# Covers ordinary equities as well as common Yahoo Finance index, futures,
# currency, crypto, and exchange suffix forms, while excluding path separators.
_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.^=_-]{0,31}$")


def normalize_ticker_symbol(value: str) -> str:
    """Return a normalized, filesystem-safe Yahoo Finance ticker symbol."""
    symbol = value.strip().upper()
    if not symbol or not _TICKER_PATTERN.fullmatch(symbol):
        raise InvalidTickerError(
            f"Ticker '{symbol or value}' contains unsupported characters."
        )
    return symbol


def ticker_symbol_is_valid(value: str) -> bool:
    try:
        normalize_ticker_symbol(value)
    except (AttributeError, InvalidTickerError):
        return False
    return True
