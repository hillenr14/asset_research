#!/usr/bin/env python3
"""Inspect brokerage XLSX/CSV exports using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DATE_COLUMNS = {"Date", "Trade Date", "Settlement Date", "Price as of Date", "Activity Date"}
NUMERIC_COLUMNS = {
    "Quantity", "Price", "Principal", "Commission / Fees", "Net Amount",
    "Principal In Local Currency", "Commission / Fees In Local Currency",
    "Net Amount In Local Currency", "Price In Local Currency", "Market Value",
    "Market Value Change", "Gain/Loss $", "Accrued Interest", "Change Price Amount",
}


def column_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - 64
    return value - 1


def read_xlsx(path: Path) -> list[list[object]]:
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                shared.append("".join(node.text or "" for node in item.iterfind(".//m:t", NS)))

        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[object]] = []
        for row_node in root.findall(".//m:sheetData/m:row", NS):
            row: list[object] = []
            for cell in row_node.findall("m:c", NS):
                idx = column_index(cell.attrib["r"])
                while len(row) <= idx:
                    row.append("")
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", NS)
                inline = cell.find("m:is", NS)
                if inline is not None:
                    value: object = "".join(node.text or "" for node in inline.iterfind(".//m:t", NS))
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared[int(value_node.text)]
                elif cell_type == "b":
                    value = value_node.text == "1"
                else:
                    raw = value_node.text or ""
                    try:
                        value = float(raw)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = raw
                row[idx] = value
            rows.append(row)
        return rows


def read_csv(path: Path) -> list[list[object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [list(row) for row in csv.reader(handle)]


def read_table(path: Path) -> list[list[object]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return read_xlsx(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_header(rows: list[list[object]], required: set[str]) -> int:
    for idx, row in enumerate(rows):
        values = {str(value).strip() for value in row}
        if required.issubset(values):
            return idx
    raise ValueError(f"Could not find header containing {sorted(required)}")


def excel_date(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        return (date(1899, 12, 30) + timedelta(days=float(value))).isoformat()
    text = str(value).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return text


def canonical_number(value: object) -> str:
    if value in (None, ""):
        return ""
    text = str(value).replace(",", "").replace("$", "").strip()
    if text.endswith("%"):
        text = str(float(text[:-1]) / 100)
    try:
        return f"{float(text):.6f}".rstrip("0").rstrip(".") or "0"
    except ValueError:
        return normalize_text(value)


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def records(rows: list[list[object]], header_index: int) -> list[dict[str, object]]:
    header = [str(value).strip() for value in rows[header_index]]
    width = len(header)
    result = []
    for row in rows[header_index + 1 :]:
        padded = list(row) + [""] * max(0, width - len(row))
        if not any(value not in (None, "") for value in padded):
            continue
        result.append(dict(zip(header, padded[:width])))
    return result


def history_fingerprint(record: dict[str, object], header: list[str]) -> str:
    parts = []
    for key in header:
        if not key:
            continue
        value = record.get(key, "")
        if key in DATE_COLUMNS:
            parts.append(excel_date(value))
        elif key in NUMERIC_COLUMNS:
            parts.append(canonical_number(value))
        else:
            parts.append(normalize_text(value))
    return "|".join(parts)


def history_summary(path: Path, existing: Path | None) -> dict[str, object]:
    rows = read_table(path)
    header_index = find_header(rows, {"Date", "Security ID", "Activity Description"})
    header = [str(value).strip() for value in rows[header_index]]
    incoming = records(rows, header_index)
    incoming_keys = [history_fingerprint(record, header) for record in incoming]
    seen: set[str] = set()
    incoming_duplicates = 0
    for key in incoming_keys:
        if key in seen:
            incoming_duplicates += 1
        seen.add(key)

    result: dict[str, object] = {
        "file": str(path),
        "header_row": header_index + 1,
        "transaction_count": len(incoming),
        "duplicate_rows_within_file": incoming_duplicates,
        "newest_transaction_date": max((excel_date(r.get("Date")) for r in incoming), default=""),
        "transactions": incoming,
    }
    if existing:
        old_rows = read_table(existing)
        old_header_index = find_header(old_rows, {"Date", "Security ID", "Activity Description"})
        old_header = [str(value).strip() for value in old_rows[old_header_index]]
        old = records(old_rows, old_header_index)
        old_keys = [history_fingerprint(record, old_header) for record in old]
        old_seen: set[str] = set()
        old_duplicates = 0
        for key in old_keys:
            if key in old_seen:
                old_duplicates += 1
            old_seen.add(key)
        overlap = sum(key in old_seen for key in incoming_keys)
        result.update({
            "existing_transaction_count": len(old),
            "existing_duplicate_rows": old_duplicates,
            "incoming_overlap_count": overlap,
            "unique_new_count": len(incoming_keys) - overlap - incoming_duplicates,
        })
    return result


def holdings_summary(path: Path) -> dict[str, object]:
    rows = read_table(path)
    header_index = find_header(rows, {"Security ID", "Quantity", "Market Value"})
    items = records(rows, header_index)
    positions = []
    for item in items:
        positions.append({
            "security_id": str(item.get("Security ID", "")).strip(),
            "cusip": str(item.get("CUSIP", "")).strip(),
            "quantity": float(canonical_number(item.get("Quantity", 0)) or 0),
            "market_value": float(canonical_number(item.get("Market Value", 0)) or 0),
            "activity_date": excel_date(item.get("Activity Date", "")),
        })
    printed_as_of = ""
    for row in rows[:header_index]:
        for value in row:
            match = re.search(r"Holdings as of\s*:\s*(\d{2}/\d{2}/\d{4})", str(value), re.I)
            if match:
                printed_as_of = excel_date(match.group(1))
    return {
        "file": str(path),
        "header_row": header_index + 1,
        "printed_as_of_date": printed_as_of,
        "position_count": len(positions),
        "total_market_value": round(sum(p["market_value"] for p in positions), 2),
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdings", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--existing-history", "--existing-history-csv", dest="existing_history", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.holdings and not args.history:
        parser.error("provide --holdings and/or --history")
    output: dict[str, object] = {}
    if args.holdings:
        output["holdings"] = holdings_summary(args.holdings)
    if args.history:
        output["history"] = history_summary(args.history, args.existing_history)
    if args.summary_only:
        for section in output.values():
            section.pop("positions", None)
            section.pop("transactions", None)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
