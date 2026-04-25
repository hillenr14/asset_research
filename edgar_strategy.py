from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
SEC_CACHE_DIR = DATA_DIR / "sec"
COMPANYFACTS_DIR = DATA_DIR / "companyfacts"
ANALYSIS_DIR = DATA_DIR / "analysis"
HISTORY_CACHE_DIR = CACHE_DIR / "history"

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}"
)

DEFAULT_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "asset_research/0.1 (local research script; set SEC_USER_AGENT with contact info)",
)
SEC_REQUEST_PAUSE_SECONDS = 0.2

BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"


@dataclass
class FilingRecord:
    ticker: str
    company: str
    cik: str
    form: str
    filing_date: str
    report_date: str
    accession_number: str
    accession_no_dashes: str
    primary_document: str
    url: str


CONCEPT_SPECS: dict[str, dict[str, Any]] = {
    "revenue": {
        "taxonomies": ["us-gaap"],
        "tags": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
        ],
        "kind": "duration",
    },
    "gross_profit": {
        "taxonomies": ["us-gaap"],
        "tags": ["GrossProfit"],
        "kind": "duration",
    },
    "operating_income": {
        "taxonomies": ["us-gaap"],
        "tags": ["OperatingIncomeLoss"],
        "kind": "duration",
    },
    "net_income": {
        "taxonomies": ["us-gaap"],
        "tags": ["NetIncomeLoss", "ProfitLoss"],
        "kind": "duration",
    },
    "operating_cash_flow": {
        "taxonomies": ["us-gaap"],
        "tags": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "kind": "duration",
    },
    "capex": {
        "taxonomies": ["us-gaap"],
        "tags": [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "CapitalExpendituresIncurredButNotYetPaid",
        ],
        "kind": "duration",
    },
    "cash_and_equivalents": {
        "taxonomies": ["us-gaap"],
        "tags": ["CashAndCashEquivalentsAtCarryingValue"],
        "kind": "instant",
    },
    "current_assets": {
        "taxonomies": ["us-gaap"],
        "tags": ["AssetsCurrent"],
        "kind": "instant",
    },
    "current_liabilities": {
        "taxonomies": ["us-gaap"],
        "tags": ["LiabilitiesCurrent"],
        "kind": "instant",
    },
    "total_assets": {
        "taxonomies": ["us-gaap"],
        "tags": ["Assets"],
        "kind": "instant",
    },
    "total_liabilities": {
        "taxonomies": ["us-gaap"],
        "tags": ["Liabilities"],
        "kind": "instant",
    },
    "stockholders_equity": {
        "taxonomies": ["us-gaap"],
        "tags": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "kind": "instant",
    },
    "long_term_debt": {
        "taxonomies": ["us-gaap"],
        "tags": [
            "LongTermDebtNoncurrent",
            "LongTermDebt",
            "LongTermDebtAndCapitalLeaseObligations",
        ],
        "kind": "instant",
    },
    "current_debt": {
        "taxonomies": ["us-gaap"],
        "tags": [
            "LongTermDebtCurrent",
            "ShortTermBorrowings",
            "ShortTermDebt",
            "CommercialPaper",
            "LongTermDebtAndCapitalLeaseObligationsCurrent",
        ],
        "kind": "instant",
    },
    "shares_outstanding": {
        "taxonomies": ["dei"],
        "tags": ["EntityCommonStockSharesOutstanding"],
        "kind": "instant",
    },
    "diluted_shares": {
        "taxonomies": ["us-gaap"],
        "tags": [
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        ],
        "kind": "duration",
    },
    "eps_diluted": {
        "taxonomies": ["us-gaap"],
        "tags": ["EarningsPerShareDiluted"],
        "kind": "duration",
    },
}


def _ensure_directories() -> None:
    for path in [DATA_DIR, CACHE_DIR, SEC_CACHE_DIR, COMPANYFACTS_DIR, ANALYSIS_DIR, HISTORY_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _sec_request(url: str, as_json: bool = True) -> Any:
    host = urlparse(url).netloc
    request = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Host": host,
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
            if response.headers.get("Content-Encoding", "").lower() == "gzip":
                payload = gzip.decompress(payload)
    except HTTPError as exc:
        message = f"SEC request failed for {url}: {exc}"
        if exc.code == 403:
            message += (
                ". SEC often requires a declared User-Agent with contact information. "
                "Set SEC_USER_AGENT, for example: 'Your Name your.email@domain.com'."
            )
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"SEC request failed for {url}: {exc}") from exc

    time.sleep(SEC_REQUEST_PAUSE_SECONDS)
    if as_json:
        return json.loads(payload.decode("utf-8"))
    return payload


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)


def _normalize_cik(value: str | int) -> str:
    return str(value).strip().lstrip("0").zfill(10)


def _ticker_map_path() -> Path:
    return SEC_CACHE_DIR / "company_tickers.json"


def load_or_fetch_ticker_map(refresh: bool = False) -> dict[str, Any]:
    path = _ticker_map_path()
    if path.exists() and not refresh:
        return _read_json(path)

    payload = _sec_request(SEC_TICKERS_URL, as_json=True)
    _write_json(path, payload)
    return payload


def resolve_ticker_to_cik(ticker: str, refresh: bool = False) -> tuple[str, str]:
    ticker = ticker.upper().strip()
    ticker_map = load_or_fetch_ticker_map(refresh=refresh)

    for entry in ticker_map.values():
        if entry.get("ticker", "").upper() == ticker:
            cik = _normalize_cik(entry["cik_str"])
            title = entry.get("title", ticker)
            return cik, title

    raise ValueError(f"Ticker not found in SEC ticker list: {ticker}")


def _columnar_filings_to_frame(filings_block: dict[str, list[Any]]) -> pd.DataFrame:
    if not filings_block:
        return pd.DataFrame()

    keys = list(filings_block.keys())
    if not keys:
        return pd.DataFrame()

    row_count = max(len(filings_block.get(key, [])) for key in keys)
    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        row = {}
        for key in keys:
            values = filings_block.get(key, [])
            row[key] = values[idx] if idx < len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _load_submissions_payload(cik: str, refresh: bool = False) -> dict[str, Any]:
    _ensure_directories()
    cache_path = SEC_CACHE_DIR / f"CIK{cik}_submissions.json"
    if cache_path.exists() and not refresh:
        return _read_json(cache_path)

    payload = _sec_request(SEC_SUBMISSIONS_URL.format(cik=cik), as_json=True)

    recent = _columnar_filings_to_frame(payload.get("filings", {}).get("recent", {}))
    extra_files = payload.get("filings", {}).get("files", [])
    frames = [recent] if not recent.empty else []

    for file_meta in extra_files:
        name = file_meta.get("name")
        if not name:
            continue
        extra_payload = _sec_request(SEC_SUBMISSIONS_BASE_URL.format(name=name), as_json=True)
        extra_frame = _columnar_filings_to_frame(extra_payload)
        if not extra_frame.empty:
            frames.append(extra_frame)

    filings_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not filings_df.empty:
        filings_df = filings_df.where(pd.notnull(filings_df), None)
        payload["filings_flat"] = filings_df.to_dict(orient="records")
    else:
        payload["filings_flat"] = []

    _write_json(cache_path, payload)
    return payload


def load_company_filings(ticker: str, refresh: bool = False) -> tuple[pd.DataFrame, str, str]:
    cik, company_name = resolve_ticker_to_cik(ticker, refresh=refresh)
    payload = _load_submissions_payload(cik, refresh=refresh)
    filings = pd.DataFrame(payload.get("filings_flat", []))
    if filings.empty:
        raise RuntimeError(f"No SEC filings found for {ticker}")

    required = ["form", "filingDate", "reportDate", "accessionNumber", "primaryDocument"]
    for column in required:
        if column not in filings.columns:
            filings[column] = None

    filings["filingDate"] = pd.to_datetime(filings["filingDate"], errors="coerce")
    filings["reportDate"] = pd.to_datetime(filings["reportDate"], errors="coerce")
    filings = filings.dropna(subset=["filingDate", "accessionNumber", "primaryDocument"])
    filings["ticker"] = ticker.upper()
    filings["company"] = payload.get("name", company_name)
    filings["cik"] = cik
    filings["accessionNoDashes"] = filings["accessionNumber"].str.replace("-", "", regex=False)
    filings["form"] = filings["form"].astype(str)
    filings = filings.sort_values(["filingDate", "accessionNumber"]).reset_index(drop=True)
    return filings, cik, payload.get("name", company_name)


def _filing_output_paths(ticker: str, form: str, filing_date: str, accession_no_dashes: str) -> tuple[Path, Path]:
    base_dir = DATA_DIR / form / ticker.upper()
    base_name = f"{filing_date}_{accession_no_dashes}"
    html_path = base_dir / f"{base_name}.html"
    json_path = base_dir / f"{base_name}.json"
    return html_path, json_path


def _select_recent_filings(filings: pd.DataFrame, years: int) -> pd.DataFrame:
    cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years))
    selected = filings[
        filings["form"].isin(["10-Q", "10-K"]) & (filings["filingDate"] >= cutoff)
    ].copy()
    return selected.sort_values(["filingDate", "form"]).reset_index(drop=True)


def download_sec_filings(ticker: str, years: int = 10, refresh: bool = False) -> list[dict[str, Any]]:
    filings, cik, company_name = load_company_filings(ticker, refresh=refresh)
    selected = _select_recent_filings(filings, years=years)
    records: list[dict[str, Any]] = []

    for filing in selected.itertuples(index=False):
        filing_date = filing.filingDate.strftime("%Y-%m-%d")
        report_date = (
            filing.reportDate.strftime("%Y-%m-%d")
            if pd.notna(filing.reportDate)
            else filing_date
        )
        html_path, json_path = _filing_output_paths(
            ticker=ticker,
            form=filing.form,
            filing_date=filing_date,
            accession_no_dashes=filing.accessionNoDashes,
        )
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{filing.accessionNoDashes}-index.html"
        )
        record = FilingRecord(
            ticker=ticker.upper(),
            company=company_name,
            cik=str(int(cik)),
            form=filing.form,
            filing_date=filing_date,
            report_date=report_date,
            accession_number=filing.accessionNumber,
            accession_no_dashes=filing.accessionNoDashes,
            primary_document=filing.primaryDocument,
            url=index_url,
        )
        record_dict = record.__dict__.copy()

        if not json_path.exists():
            _write_json(json_path, record_dict)

        if not html_path.exists():
            filing_url = SEC_ARCHIVE_URL.format(
                cik=int(cik),
                accession_no_dashes=filing.accessionNoDashes,
                primary_document=filing.primaryDocument,
            )
            payload = _sec_request(filing_url, as_json=False)
            _write_bytes(html_path, payload)

        records.append(record_dict)

    return records


def load_or_fetch_companyfacts(
    ticker: str,
    cik: str,
    refresh: bool = False,
) -> dict[str, Any]:
    _ensure_directories()
    path = COMPANYFACTS_DIR / f"{ticker.upper()}.json"
    if path.exists() and not refresh:
        return _read_json(path)

    payload = _sec_request(SEC_COMPANYFACTS_URL.format(cik=cik), as_json=True)
    _write_json(path, payload)
    return payload


def _normalize_fact_df(df: pd.DataFrame, concept: str, taxonomy: str, tag: str, unit: str) -> pd.DataFrame:
    normalized = df.copy()
    for column in ["start", "end", "filed"]:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    normalized["concept"] = concept
    normalized["taxonomy"] = taxonomy
    normalized["tag"] = tag
    normalized["unit"] = unit
    if "val" in normalized.columns:
        normalized["val"] = pd.to_numeric(normalized["val"], errors="coerce")
    if "fy" in normalized.columns:
        normalized["fy"] = pd.to_numeric(normalized["fy"], errors="coerce")
    if "start" in normalized.columns and "end" in normalized.columns:
        normalized["days"] = (normalized["end"] - normalized["start"]).dt.days
    else:
        normalized["days"] = np.nan
    normalized = normalized.dropna(subset=["val", "end"])
    for column in ["form", "fp", "accn", "frame"]:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized


def _extract_best_fact_series(companyfacts: dict[str, Any], concept: str) -> pd.DataFrame:
    spec = CONCEPT_SPECS[concept]
    best_df = pd.DataFrame()

    for taxonomy in spec["taxonomies"]:
        taxonomy_block = companyfacts.get("facts", {}).get(taxonomy, {})
        for tag in spec["tags"]:
            tag_block = taxonomy_block.get(tag, {})
            units = tag_block.get("units", {})
            for unit_name, observations in units.items():
                candidate = pd.DataFrame(observations)
                if candidate.empty:
                    continue
                candidate = _normalize_fact_df(candidate, concept, taxonomy, tag, unit_name)
                if candidate.empty:
                    continue
                if spec["kind"] == "instant":
                    if "start" in candidate.columns:
                        candidate = candidate[candidate["start"].isna() | (candidate["start"] == candidate["end"])]
                else:
                    candidate = candidate[candidate["start"].notna()]
                if len(candidate) > len(best_df):
                    best_df = candidate.copy()

    if best_df.empty:
        return best_df

    best_df = best_df.sort_values(["end", "filed", "accn"]).reset_index(drop=True)
    return best_df


def _safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(denominator, pd.Series):
        return numerator / denominator.replace({0: np.nan})
    if denominator == 0:
        denominator = np.nan
    return numerator / denominator


def _deduplicate_periods(df: pd.DataFrame, date_col: str = "end") -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values([date_col, "filed", "accn"]).copy()
    return ordered.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)


def _build_duration_annual_series(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    annual = raw_df[(raw_df["form"] == "10-K") & (raw_df["days"] >= 300)].copy()
    return _deduplicate_periods(annual)


def _build_instant_quarter_series(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    allowed_forms = raw_df["form"].isin(["10-Q", "10-K"])
    quarter = raw_df[allowed_forms].copy()
    return _deduplicate_periods(quarter)


def _build_instant_annual_series(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    annual = raw_df[raw_df["form"] == "10-K"].copy()
    return _deduplicate_periods(annual)


def _pick_fiscal_year(record: pd.Series) -> int | None:
    if pd.notna(record.get("fy")):
        return int(record["fy"])
    if pd.notna(record.get("end")):
        return int(pd.Timestamp(record["end"]).year)
    return None


def _build_duration_quarter_series(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    frame = raw_df[raw_df["form"].isin(["10-Q", "10-K"])].copy()
    if frame.empty:
        return frame

    frame["fiscal_year"] = frame.apply(_pick_fiscal_year, axis=1)
    frame = frame.dropna(subset=["fiscal_year"])
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)

    records: list[dict[str, Any]] = []

    for fiscal_year, group in frame.groupby("fiscal_year"):
        annual = group[(group["form"] == "10-K") & (group["days"] >= 300)].sort_values(["filed", "end"])
        q1 = group[group["fp"] == "Q1"].sort_values(["filed", "end"])
        q2 = group[group["fp"] == "Q2"].sort_values(["filed", "end"])
        q3 = group[group["fp"] == "Q3"].sort_values(["filed", "end"])

        q1_row = q1.iloc[-1] if not q1.empty else None
        q2_row = q2.iloc[-1] if not q2.empty else None
        q3_row = q3.iloc[-1] if not q3.empty else None
        annual_row = annual.iloc[-1] if not annual.empty else None

        q1_value = None
        q2_value = None
        q3_value = None

        if q1_row is not None:
            q1_value = float(q1_row["val"])
            records.append({**q1_row.to_dict(), "quarter": "Q1", "single_quarter_val": q1_value})

        if q2_row is not None:
            if q1_value is not None and pd.notna(q2_row.get("val")):
                q2_value = float(q2_row["val"]) - q1_value
            elif pd.notna(q2_row.get("days")) and q2_row["days"] <= 120:
                q2_value = float(q2_row["val"])
            if q2_value is not None:
                records.append({**q2_row.to_dict(), "quarter": "Q2", "single_quarter_val": q2_value})

        if q3_row is not None:
            if q2_row is not None and pd.notna(q3_row.get("val")) and pd.notna(q2_row.get("val")):
                q3_value = float(q3_row["val"]) - float(q2_row["val"])
            elif pd.notna(q3_row.get("days")) and q3_row["days"] <= 120:
                q3_value = float(q3_row["val"])
            if q3_value is not None:
                records.append({**q3_row.to_dict(), "quarter": "Q3", "single_quarter_val": q3_value})

        if annual_row is not None:
            components = [value for value in [q1_value, q2_value, q3_value] if value is not None]
            if len(components) == 3 and pd.notna(annual_row.get("val")):
                q4_value = float(annual_row["val"]) - float(sum(components))
                records.append({**annual_row.to_dict(), "quarter": "Q4", "single_quarter_val": q4_value})

    quarter_df = pd.DataFrame(records)
    if quarter_df.empty:
        return quarter_df

    quarter_df["val"] = quarter_df["single_quarter_val"]
    quarter_df = quarter_df.drop(columns=["single_quarter_val"])
    quarter_df = quarter_df.sort_values(["end", "filed", "accn"])
    quarter_df = quarter_df.drop_duplicates(subset=["end"], keep="last").reset_index(drop=True)
    return quarter_df


def _prepare_filing_lookup(filing_records: list[dict[str, Any]]) -> pd.DataFrame:
    filings = pd.DataFrame(filing_records)
    if filings.empty:
        return filings
    filings["report_date"] = pd.to_datetime(filings["report_date"], errors="coerce")
    filings["filing_date"] = pd.to_datetime(filings["filing_date"], errors="coerce")
    filings = filings.dropna(subset=["report_date", "filing_date"])
    return filings.sort_values(["report_date", "filing_date", "form"])


def _attach_filing_dates(frame: pd.DataFrame, filing_lookup: pd.DataFrame, annual: bool) -> pd.DataFrame:
    if frame.empty:
        return frame

    output = frame.copy()
    output["period_end"] = pd.to_datetime(output["end"], errors="coerce")
    if not filing_lookup.empty:
        forms = ["10-K"] if annual else ["10-Q", "10-K"]
        subset = filing_lookup[filing_lookup["form"].isin(forms)][["report_date", "filing_date", "form"]]
        subset = subset.sort_values(["report_date", "filing_date"])
        output = output.merge(
            subset,
            how="left",
            left_on="period_end",
            right_on="report_date",
        )
        output["filing_date"] = output["filing_date"].fillna(output["filed"])
        output = output.drop(columns=["report_date"])
    else:
        output["filing_date"] = output["filed"]
    return output


def build_financial_datasets(
    companyfacts: dict[str, Any],
    filing_records: list[dict[str, Any]],
    years: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filing_lookup = _prepare_filing_lookup(filing_records)
    cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years))

    quarter_series: dict[str, pd.DataFrame] = {}
    annual_series: dict[str, pd.DataFrame] = {}

    for concept in CONCEPT_SPECS:
        raw_series = _extract_best_fact_series(companyfacts, concept)
        if raw_series.empty:
            continue
        raw_series = raw_series[raw_series["end"] >= cutoff].copy()
        if raw_series.empty:
            continue

        if CONCEPT_SPECS[concept]["kind"] == "duration":
            quarter_df = _build_duration_quarter_series(raw_series)
            annual_df = _build_duration_annual_series(raw_series)
        else:
            quarter_df = _build_instant_quarter_series(raw_series)
            annual_df = _build_instant_annual_series(raw_series)

        quarter_df = _attach_filing_dates(quarter_df, filing_lookup, annual=False)
        annual_df = _attach_filing_dates(annual_df, filing_lookup, annual=True)

        if not quarter_df.empty:
            quarter_series[concept] = quarter_df[["period_end", "filing_date", "val"]].rename(columns={"val": concept})
        if not annual_df.empty:
            annual_series[concept] = annual_df[["period_end", "filing_date", "val"]].rename(columns={"val": concept})

    quarterly_df = None
    for concept, series in quarter_series.items():
        quarterly_df = series if quarterly_df is None else quarterly_df.merge(series, on=["period_end", "filing_date"], how="outer")

    annual_df = None
    for concept, series in annual_series.items():
        annual_df = series if annual_df is None else annual_df.merge(series, on=["period_end", "filing_date"], how="outer")

    quarterly = quarterly_df if quarterly_df is not None else pd.DataFrame(columns=["period_end", "filing_date"])
    annual = annual_df if annual_df is not None else pd.DataFrame(columns=["period_end", "filing_date"])

    quarterly["period_end"] = pd.to_datetime(quarterly["period_end"], errors="coerce")
    quarterly["filing_date"] = pd.to_datetime(quarterly["filing_date"], errors="coerce")
    quarterly = quarterly.sort_values(["period_end", "filing_date"]).drop_duplicates(subset=["period_end"], keep="last")

    annual["period_end"] = pd.to_datetime(annual["period_end"], errors="coerce")
    annual["filing_date"] = pd.to_datetime(annual["filing_date"], errors="coerce")
    annual = annual.sort_values(["period_end", "filing_date"]).drop_duplicates(subset=["period_end"], keep="last")

    return quarterly.reset_index(drop=True), annual.reset_index(drop=True)


def compute_financial_indicators(
    quarterly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quarterly = quarterly_df.copy()
    annual = annual_df.copy()

    if quarterly.empty:
        raise RuntimeError("Quarterly financial dataset is empty after SEC extraction.")

    for frame in [quarterly, annual]:
        frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
        frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
        frame.sort_values("period_end", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        for column in frame.columns:
            if column not in {"period_end", "filing_date"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "current_debt" not in quarterly.columns:
        quarterly["current_debt"] = np.nan
    if "long_term_debt" not in quarterly.columns:
        quarterly["long_term_debt"] = np.nan
    quarterly["total_debt"] = quarterly[["current_debt", "long_term_debt"]].sum(axis=1, min_count=1)

    if "current_debt" not in annual.columns:
        annual["current_debt"] = np.nan
    if "long_term_debt" not in annual.columns:
        annual["long_term_debt"] = np.nan
    annual["total_debt"] = annual[["current_debt", "long_term_debt"]].sum(axis=1, min_count=1)

    if "capex" in quarterly.columns:
        quarterly["capex_abs"] = quarterly["capex"].abs()
    else:
        quarterly["capex_abs"] = np.nan

    if "capex" in annual.columns:
        annual["capex_abs"] = annual["capex"].abs()
    else:
        annual["capex_abs"] = np.nan

    quarterly["free_cash_flow"] = quarterly.get("operating_cash_flow", np.nan) - quarterly["capex_abs"]
    annual["free_cash_flow"] = annual.get("operating_cash_flow", np.nan) - annual["capex_abs"]

    for metric in ["revenue", "net_income", "eps_diluted", "operating_cash_flow", "free_cash_flow"]:
        if metric in quarterly.columns:
            quarterly[f"{metric}_ttm"] = quarterly[metric].rolling(4, min_periods=4).sum()
            quarterly[f"{metric}_growth_yoy"] = quarterly[metric].pct_change(4, fill_method=None)

    if "gross_profit" in quarterly.columns and "revenue" in quarterly.columns:
        quarterly["gross_margin"] = _safe_divide(quarterly["gross_profit"], quarterly["revenue"])
    else:
        quarterly["gross_margin"] = np.nan

    if "operating_income" in quarterly.columns and "revenue" in quarterly.columns:
        quarterly["operating_margin"] = _safe_divide(quarterly["operating_income"], quarterly["revenue"])
    else:
        quarterly["operating_margin"] = np.nan

    if "net_income" in quarterly.columns and "revenue" in quarterly.columns:
        quarterly["net_margin"] = _safe_divide(quarterly["net_income"], quarterly["revenue"])
    else:
        quarterly["net_margin"] = np.nan

    quarterly["current_ratio"] = _safe_divide(
        quarterly.get("current_assets", np.nan),
        quarterly.get("current_liabilities", pd.Series(np.nan, index=quarterly.index)),
    )
    quarterly["debt_to_equity"] = _safe_divide(
        quarterly.get("total_debt", np.nan),
        quarterly.get("stockholders_equity", pd.Series(np.nan, index=quarterly.index)),
    )
    quarterly["roa"] = _safe_divide(
        quarterly.get("net_income_ttm", np.nan),
        quarterly.get("total_assets", pd.Series(np.nan, index=quarterly.index)),
    )
    quarterly["roe"] = _safe_divide(
        quarterly.get("net_income_ttm", np.nan),
        quarterly.get("stockholders_equity", pd.Series(np.nan, index=quarterly.index)),
    )
    quarterly["asset_turnover"] = _safe_divide(
        quarterly.get("revenue_ttm", np.nan),
        quarterly.get("total_assets", pd.Series(np.nan, index=quarterly.index)),
    )
    quarterly["operating_cash_flow_margin"] = _safe_divide(
        quarterly.get("operating_cash_flow_ttm", np.nan),
        quarterly.get("revenue_ttm", pd.Series(np.nan, index=quarterly.index)),
    )
    quarterly["free_cash_flow_margin"] = _safe_divide(
        quarterly.get("free_cash_flow_ttm", np.nan),
        quarterly.get("revenue_ttm", pd.Series(np.nan, index=quarterly.index)),
    )
    quarterly["accrual_ratio"] = _safe_divide(
        quarterly.get("net_income_ttm", np.nan) - quarterly.get("operating_cash_flow_ttm", np.nan),
        quarterly.get("total_assets", pd.Series(np.nan, index=quarterly.index)),
    )

    if not annual.empty:
        for metric in ["revenue", "net_income", "eps_diluted", "operating_cash_flow", "free_cash_flow"]:
            if metric in annual.columns:
                annual[f"{metric}_growth_yoy"] = annual[metric].pct_change(1, fill_method=None)
        if "net_income" in annual.columns and "stockholders_equity" in annual.columns:
            annual["roe"] = _safe_divide(annual["net_income"], annual["stockholders_equity"])
        else:
            annual["roe"] = np.nan
        if "net_income" in annual.columns and "total_assets" in annual.columns:
            annual["roa"] = _safe_divide(annual["net_income"], annual["total_assets"])
        else:
            annual["roa"] = np.nan

    annual_signals = annual[
        [col for col in annual.columns if col in {"period_end", "filing_date", "revenue_growth_yoy", "eps_diluted_growth_yoy", "free_cash_flow_growth_yoy", "roe", "roa"}]
    ].copy()

    if not annual_signals.empty:
        annual_signals = annual_signals.rename(
            columns={
                "filing_date": "annual_filing_date",
                "revenue_growth_yoy": "annual_revenue_growth",
                "eps_diluted_growth_yoy": "annual_eps_growth",
                "free_cash_flow_growth_yoy": "annual_fcf_growth",
                "roe": "annual_roe",
                "roa": "annual_roa",
            }
        )
        quarterly = pd.merge_asof(
            quarterly.sort_values("period_end"),
            annual_signals.sort_values("period_end"),
            on="period_end",
            direction="backward",
        )

    return quarterly, annual


def _history_cache_path(ticker: str) -> Path:
    HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_CACHE_DIR / f"{ticker.upper()}.pkl"


def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized


def _read_history_cache(ticker: str) -> pd.DataFrame:
    path = _history_cache_path(ticker)
    if not path.exists():
        return pd.DataFrame()
    try:
        return _normalize_history_df(pd.read_pickle(path))
    except Exception:
        return pd.DataFrame()


def _write_history_cache(ticker: str, df: pd.DataFrame) -> None:
    _normalize_history_df(df).to_pickle(_history_cache_path(ticker))


def _download_history_range(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if start_date >= end_date:
        return pd.DataFrame()
    history = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)
    return _normalize_history_df(history)


def download_price_history(ticker: str, start_date: pd.Timestamp) -> pd.DataFrame:
    end_date = pd.Timestamp(date.today()).normalize()
    cached = _read_history_cache(ticker)
    pieces = [cached] if not cached.empty else []

    if cached.empty:
        fetched = _download_history_range(ticker, start_date, end_date + pd.Timedelta(days=1))
        pieces.append(fetched)
    else:
        cached_start = cached.index.min().normalize()
        cached_end = cached.index.max().normalize()
        if start_date < cached_start:
            pieces.append(_download_history_range(ticker, start_date, cached_start))
        if end_date > cached_end:
            pieces.append(
                _download_history_range(
                    ticker,
                    cached_end + pd.Timedelta(days=1),
                    end_date + pd.Timedelta(days=1),
                )
            )

    if not pieces:
        return pd.DataFrame()

    merged = _normalize_history_df(pd.concat(pieces))
    _write_history_cache(ticker, merged)
    return merged.loc[merged.index >= start_date].copy()


def _next_trading_day(index: pd.DatetimeIndex, target_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    pos = index.searchsorted(target_date, side="right")
    if pos >= len(index):
        return pd.NaT
    return index[pos]


def _score_positive(value: float | None, buy_min: float, sell_max: float) -> int:
    if value is None or pd.isna(value):
        return 0
    if value >= buy_min:
        return 1
    if value <= sell_max:
        return -1
    return 0


def _score_negative_better(value: float | None, buy_max: float, sell_min: float) -> int:
    if value is None or pd.isna(value):
        return 0
    if value <= buy_max:
        return 1
    if value >= sell_min:
        return -1
    return 0


def _series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def generate_signals(indicators_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if indicators_df.empty:
        raise RuntimeError("Indicators dataset is empty.")
    if price_df.empty:
        raise RuntimeError("Price history dataset is empty.")

    signals = indicators_df.copy().sort_values("filing_date").reset_index(drop=True)
    signals["effective_date"] = signals["filing_date"].apply(lambda d: _next_trading_day(price_df.index, pd.Timestamp(d)))
    signals = signals.dropna(subset=["effective_date"]).copy()

    price_lookup = price_df["Adj Close"] if "Adj Close" in price_df.columns else price_df["Close"]
    signals["price"] = signals["effective_date"].map(price_lookup)

    shares = _series_or_nan(signals, "shares_outstanding")
    revenue_ttm = _series_or_nan(signals, "revenue_ttm")
    eps_ttm = _series_or_nan(signals, "eps_diluted_ttm")
    fcf_ttm = _series_or_nan(signals, "free_cash_flow_ttm")

    signals["sales_per_share_ttm"] = _safe_divide(revenue_ttm, shares)
    signals["price_to_sales"] = _safe_divide(signals["price"], signals["sales_per_share_ttm"])
    signals["price_to_earnings"] = _safe_divide(signals["price"], eps_ttm)
    signals["market_cap"] = signals["price"] * shares
    signals["free_cash_flow_yield"] = _safe_divide(fcf_ttm, signals["market_cap"])
    signals["ps_rolling_median"] = signals["price_to_sales"].rolling(8, min_periods=4).median()
    signals["pe_rolling_median"] = signals["price_to_earnings"].rolling(8, min_periods=4).median()

    component_columns = {
        "score_growth_revenue": _series_or_nan(signals, "revenue_growth_yoy").apply(lambda x: _score_positive(x, 0.05, -0.02)),
        "score_growth_eps": _series_or_nan(signals, "eps_diluted_growth_yoy").apply(lambda x: _score_positive(x, 0.05, -0.02)),
        "score_growth_fcf": _series_or_nan(signals, "free_cash_flow_growth_yoy").apply(lambda x: _score_positive(x, 0.05, -0.05)),
        "score_margin_gross": _series_or_nan(signals, "gross_margin").apply(lambda x: _score_positive(x, 0.35, 0.15)),
        "score_margin_operating": _series_or_nan(signals, "operating_margin").apply(lambda x: _score_positive(x, 0.10, 0.03)),
        "score_profitability_roe": _series_or_nan(signals, "roe").apply(lambda x: _score_positive(x, 0.10, 0.03)),
        "score_liquidity": _series_or_nan(signals, "current_ratio").apply(lambda x: _score_positive(x, 1.0, 0.75)),
        "score_leverage": _series_or_nan(signals, "debt_to_equity").apply(lambda x: _score_negative_better(x, 1.5, 2.5)),
        "score_cash_generation": _series_or_nan(signals, "free_cash_flow_ttm").apply(lambda x: _score_positive(x, 0.0, -1.0)),
        "score_accruals": _series_or_nan(signals, "accrual_ratio").apply(lambda x: _score_negative_better(x, 0.05, 0.10)),
        "score_annual_revenue": _series_or_nan(signals, "annual_revenue_growth").apply(lambda x: _score_positive(x, 0.05, -0.02)),
        "score_annual_eps": _series_or_nan(signals, "annual_eps_growth").apply(lambda x: _score_positive(x, 0.05, -0.02)),
        "score_valuation_pe": (
            (
                (signals["price_to_earnings"] > 0)
                & (signals["pe_rolling_median"] > 0)
                & (signals["price_to_earnings"] <= signals["pe_rolling_median"] * 0.9)
            ).astype(int)
            - (
                (signals["price_to_earnings"] > 0)
                & (signals["pe_rolling_median"] > 0)
                & (signals["price_to_earnings"] >= signals["pe_rolling_median"] * 1.25)
            ).astype(int)
        ),
        "score_valuation_ps": (
            (
                (signals["price_to_sales"] > 0)
                & (signals["ps_rolling_median"] > 0)
                & (signals["price_to_sales"] <= signals["ps_rolling_median"] * 0.9)
            ).astype(int)
            - (
                (signals["price_to_sales"] > 0)
                & (signals["ps_rolling_median"] > 0)
                & (signals["price_to_sales"] >= signals["ps_rolling_median"] * 1.25)
            ).astype(int)
        ),
        "score_fcf_yield": _series_or_nan(signals, "free_cash_flow_yield").apply(lambda x: _score_positive(x, 0.04, 0.01)),
    }

    for column, values in component_columns.items():
        signals[column] = values

    score_cols = [column for column in signals.columns if column.startswith("score_")]
    signals["composite_score"] = signals[score_cols].sum(axis=1)
    signals["signal"] = np.select(
        [signals["composite_score"] >= 5, signals["composite_score"] <= -3],
        [BUY, SELL],
        default=HOLD,
    )
    signals["position"] = signals["signal"].map({BUY: 1.0, HOLD: 0.5, SELL: 0.0})

    return signals


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _compute_cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty or len(equity_curve) < 2:
        return float("nan")
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    ending_value = equity_curve.iloc[-1]
    if ending_value <= 0:
        return float("nan")
    return float(ending_value ** (1 / years) - 1)


def _period_win_rate(backtest_df: pd.DataFrame, signal_df: pd.DataFrame) -> float:
    events = signal_df[["effective_date", "signal"]].drop_duplicates().sort_values("effective_date").reset_index(drop=True)
    if events.empty:
        return float("nan")

    period_returns: list[float] = []
    for idx, event in events.iterrows():
        start = pd.Timestamp(event["effective_date"])
        end = (
            pd.Timestamp(events.iloc[idx + 1]["effective_date"])
            if idx + 1 < len(events)
            else backtest_df.index[-1]
        )
        segment = backtest_df.loc[(backtest_df.index >= start) & (backtest_df.index < end)]
        if segment.empty:
            continue
        period_returns.append(float((1 + segment["strategy_return"]).prod() - 1))

    if not period_returns:
        return float("nan")
    return float(np.mean(np.array(period_returns) > 0))


def run_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, float]]:
    backtest = price_df.copy()
    price_col = "Adj Close" if "Adj Close" in backtest.columns else "Close"
    backtest = backtest[[price_col]].rename(columns={price_col: "price"})
    backtest["asset_return"] = backtest["price"].pct_change().fillna(0.0)

    daily_positions = signal_df[["effective_date", "position", "signal", "composite_score"]].copy()
    daily_positions = daily_positions.drop_duplicates(subset=["effective_date"], keep="last")
    daily_positions["effective_date"] = pd.to_datetime(daily_positions["effective_date"], errors="coerce")
    daily_positions = daily_positions.set_index("effective_date").sort_index()

    backtest = backtest.join(daily_positions, how="left")
    backtest["position"] = backtest["position"].ffill().fillna(0.0)
    backtest["signal"] = backtest["signal"].ffill()
    backtest["composite_score"] = backtest["composite_score"].ffill()

    backtest["turnover"] = backtest["position"].diff().abs().fillna(backtest["position"].abs())
    backtest["strategy_return_gross"] = backtest["asset_return"] * backtest["position"].shift(1).fillna(0.0)
    backtest["transaction_cost"] = backtest["turnover"] * transaction_cost
    backtest["strategy_return"] = backtest["strategy_return_gross"] - backtest["transaction_cost"]

    backtest["benchmark_equity"] = (1 + backtest["asset_return"]).cumprod()
    backtest["strategy_equity"] = (1 + backtest["strategy_return"]).cumprod()

    strategy_vol = float(backtest["strategy_return"].std() * math.sqrt(252))
    strategy_return = float(backtest["strategy_return"].mean() * 252)
    benchmark_return = float(backtest["asset_return"].mean() * 252)
    sharpe = float(strategy_return / strategy_vol) if strategy_vol > 0 else float("nan")

    summary = {
        "strategy_total_return": float(backtest["strategy_equity"].iloc[-1] - 1.0),
        "benchmark_total_return": float(backtest["benchmark_equity"].iloc[-1] - 1.0),
        "strategy_cagr": _compute_cagr(backtest["strategy_equity"]),
        "benchmark_cagr": _compute_cagr(backtest["benchmark_equity"]),
        "strategy_annualized_volatility": strategy_vol,
        "benchmark_annualized_return": benchmark_return,
        "strategy_annualized_return": strategy_return,
        "strategy_sharpe": sharpe,
        "strategy_max_drawdown": _compute_max_drawdown(backtest["strategy_equity"]),
        "benchmark_max_drawdown": _compute_max_drawdown(backtest["benchmark_equity"]),
        "trade_count": int((backtest["turnover"] > 0).sum()),
        "signal_period_win_rate": _period_win_rate(backtest, signal_df),
    }

    return backtest, summary


def _serialize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in summary.items():
        if isinstance(value, (np.floating, float)):
            serialized[key] = None if (pd.isna(value) or np.isinf(value)) else float(value)
        elif isinstance(value, (np.integer, int)):
            serialized[key] = int(value)
        else:
            serialized[key] = value
    return serialized


def _save_outputs(
    ticker: str,
    quarterly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ticker = ticker.upper()

    quarterly_path = ANALYSIS_DIR / f"{ticker}_financials_quarterly.csv"
    annual_path = ANALYSIS_DIR / f"{ticker}_financials_annual.csv"
    signals_path = ANALYSIS_DIR / f"{ticker}_signals.csv"
    backtest_path = ANALYSIS_DIR / f"{ticker}_backtest.csv"
    summary_path = ANALYSIS_DIR / f"{ticker}_summary.json"

    quarterly_df.to_csv(quarterly_path, index=False)
    annual_df.to_csv(annual_path, index=False)
    signal_df.to_csv(signals_path, index=False)
    backtest_df.to_csv(backtest_path)
    _write_json(summary_path, _serialize_summary(summary))

    return {
        "quarterly_financials": str(quarterly_path),
        "annual_financials": str(annual_path),
        "signals": str(signals_path),
        "backtest": str(backtest_path),
        "summary": str(summary_path),
    }


def run_edgar_strategy(
    ticker: str,
    years: int = 10,
    refresh_sec: bool = False,
) -> dict[str, Any]:
    _ensure_directories()
    ticker = ticker.upper().strip()

    filings = download_sec_filings(ticker=ticker, years=years, refresh=refresh_sec)
    cik, _ = resolve_ticker_to_cik(ticker, refresh=refresh_sec)
    companyfacts = load_or_fetch_companyfacts(ticker=ticker, cik=cik, refresh=refresh_sec)

    quarterly_df, annual_df = build_financial_datasets(
        companyfacts=companyfacts,
        filing_records=filings,
        years=years,
    )
    quarterly_indicators, annual_indicators = compute_financial_indicators(quarterly_df, annual_df)

    price_start = pd.Timestamp(date.today() - timedelta(days=365 * years + 30))
    price_df = download_price_history(ticker=ticker, start_date=price_start)
    signal_df = generate_signals(quarterly_indicators, price_df)
    backtest_df, summary = run_backtest(signal_df, price_df)

    output_paths = _save_outputs(
        ticker=ticker,
        quarterly_df=quarterly_indicators,
        annual_df=annual_indicators,
        signal_df=signal_df,
        backtest_df=backtest_df,
        summary=summary,
    )

    return {
        "ticker": ticker,
        "filings_downloaded": len(filings),
        "quarterly_periods": int(len(quarterly_indicators)),
        "annual_periods": int(len(annual_indicators)),
        "summary": _serialize_summary(summary),
        "output_paths": output_paths,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SEC filings, compute fundamental indicators, generate signals, and backtest them.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--years", type=int, default=10, help="Trailing number of years to analyze")
    parser.add_argument(
        "--refresh-sec",
        action="store_true",
        help="Refresh SEC submissions/companyfacts caches instead of reusing cached JSON",
    )
    args = parser.parse_args()

    result = run_edgar_strategy(
        ticker=args.ticker,
        years=args.years,
        refresh_sec=args.refresh_sec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
