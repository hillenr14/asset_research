import pandas as pd

from edgar_strategy import _build_duration_quarter_series, build_financial_datasets


def _duration_fact(
    value: float,
    start: str,
    end: str,
    *,
    fp: str,
    form: str = "10-Q",
    filed: str = "2025-01-01",
    frame: str | None = None,
    concept: str = "revenue",
    accession: str = "0001",
) -> dict[str, object]:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    return {
        "val": value,
        "start": start_date,
        "end": end_date,
        "filed": pd.Timestamp(filed),
        "fp": fp,
        "form": form,
        "fy": 2024,
        "frame": frame,
        "concept": concept,
        "accn": accession,
        "days": (end_date - start_date).days,
    }


def test_duration_quarters_prefer_discrete_facts_and_derive_from_cumulative_facts() -> None:
    raw = pd.DataFrame(
        [
            _duration_fact(100, "2024-01-01", "2024-03-31", fp="Q1", frame="CY2024Q1"),
            _duration_fact(60, "2024-04-01", "2024-06-30", fp="Q2"),
            _duration_fact(160, "2024-01-01", "2024-06-30", fp="Q2", accession="0002"),
            _duration_fact(250, "2024-01-01", "2024-09-30", fp="Q3"),
            _duration_fact(370, "2024-01-01", "2024-12-31", fp="FY", form="10-K"),
        ]
    )

    result = _build_duration_quarter_series(raw).set_index("quarter")

    assert result.loc["Q1", "val"] == 100
    assert result.loc["Q2", "val"] == 60
    assert result.loc["Q3", "val"] == 90
    assert result.loc["Q4", "val"] == 120


def test_eps_is_never_derived_by_subtracting_cumulative_or_annual_values() -> None:
    raw = pd.DataFrame(
        [
            _duration_fact(
                1.0,
                "2024-01-01",
                "2024-03-31",
                fp="Q1",
                frame="CY2024Q1",
                concept="eps_diluted",
            ),
            _duration_fact(
                2.2,
                "2024-01-01",
                "2024-06-30",
                fp="Q2",
                concept="eps_diluted",
            ),
            _duration_fact(
                0.7,
                "2024-04-01",
                "2024-06-30",
                fp="Q2",
                frame="CY2024Q2",
                concept="eps_diluted",
                accession="0002",
            ),
            _duration_fact(
                3.1,
                "2024-01-01",
                "2024-09-30",
                fp="Q3",
                concept="eps_diluted",
            ),
            _duration_fact(
                4.0,
                "2024-01-01",
                "2024-12-31",
                fp="FY",
                form="10-K",
                concept="eps_diluted",
            ),
        ]
    )

    result = _build_duration_quarter_series(raw)

    assert result["quarter"].tolist() == ["Q1", "Q2"]
    assert result.set_index("quarter")["val"].to_dict() == {"Q1": 1.0, "Q2": 0.7}


def test_financial_dataset_preserves_concepts_with_different_filing_dates() -> None:
    companyfacts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 100,
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "filed": "2024-05-01",
                                "form": "10-Q",
                                "fp": "Q1",
                                "fy": 2024,
                                "frame": "CY2024Q1",
                                "accn": "0001",
                            }
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 20,
                                "start": "2024-01-01",
                                "end": "2024-03-31",
                                "filed": "2024-05-05",
                                "form": "10-Q",
                                "fp": "Q1",
                                "fy": 2024,
                                "frame": "CY2024Q1",
                                "accn": "0002",
                            }
                        ]
                    }
                },
            }
        }
    }

    quarterly, _ = build_financial_datasets(companyfacts, filing_records=[], years=10)

    assert len(quarterly) == 1
    assert quarterly.iloc[0]["revenue"] == 100
    assert quarterly.iloc[0]["net_income"] == 20
    assert quarterly.iloc[0]["filing_date"] == pd.Timestamp("2024-05-05")
