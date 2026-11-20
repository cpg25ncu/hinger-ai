"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import csv
import platform
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"


def write_csv(name: str, rows: list[dict]) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")
    return path


def read_csv(name: str) -> list[dict]:
    with (TABLES / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def machine() -> str:
    return f"{platform.processor() or platform.machine()}, Python {platform.python_version()}"
