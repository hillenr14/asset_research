from __future__ import annotations

from models import AnalysisIssue


class AnalysisAppError(Exception):
    category = "provider_error"

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


class ValidationError(AnalysisAppError):
    category = "validation_error"


class InvalidTickerError(AnalysisAppError):
    category = "invalid_ticker"


class UnsupportedAnalysisError(AnalysisAppError):
    category = "unsupported_analysis"


class MissingDataError(AnalysisAppError):
    category = "missing_data"


class ProviderError(AnalysisAppError):
    category = "provider_error"


def to_issue(exc: Exception) -> AnalysisIssue:
    if isinstance(exc, AnalysisAppError):
        return AnalysisIssue(category=exc.category, message=exc.message, details=exc.details)
    return AnalysisIssue(
        category="unexpected_error",
        message="Unexpected error during analysis.",
        details=str(exc),
    )
