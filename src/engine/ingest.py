"""Ingestion boundary for the UFCStats CSV mirror (Greco1899/scrape_ufc_stats).

Deep-module interface (stable): ``fetch_snapshot`` -> files on disk, ``load_snapshot`` -> verbatim
DataFrames, ``validate_schema`` -> fail loudly on upstream drift. No cleaning happens here:
Bronze is verbatim, and all parsing lives in Silver (``medallion.py``).
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd

SOURCE_SYSTEM = "ufcstats_via_greco1899"
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/master/"

RAW_FILES: Dict[str, str] = {
    "fight_results": "ufc_fight_results.csv",
    "fight_stats": "ufc_fight_stats.csv",
    "fighters": "ufc_fighter_tott.csv",
    "events": "ufc_event_details.csv",
}

EXPECTED_COLUMNS: Dict[str, List[str]] = {
    "fight_results": ["EVENT", "BOUT", "OUTCOME", "WEIGHTCLASS", "METHOD", "ROUND", "TIME", "TIME FORMAT",
                      "REFEREE", "DETAILS", "URL"],
    "fight_stats": ["EVENT", "BOUT", "ROUND", "FIGHTER", "KD", "SIG.STR.", "SIG.STR. %", "TOTAL STR.", "TD",
                    "TD %", "SUB.ATT", "REV.", "CTRL", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"],
    "fighters": ["FIGHTER", "HEIGHT", "WEIGHT", "REACH", "STANCE", "DOB", "URL"],
    "events": ["EVENT", "URL", "DATE", "LOCATION"],
}


class SchemaDriftError(ValueError):
    """Raised when an upstream file is empty or no longer matches the expected column contract."""


def validate_schema(name: str, df: pd.DataFrame) -> None:
    """Check a raw frame against the column contract.

    Args:
        name: Logical source name (a key of ``RAW_FILES``).
        df: The raw DataFrame.

    Raises:
        SchemaDriftError: If the frame is empty or any expected column is missing. Extra columns are allowed.
    """
    if name not in EXPECTED_COLUMNS:
        raise SchemaDriftError(f"Unknown source '{name}'")
    if df.empty:
        raise SchemaDriftError(f"{name}: frame is empty")
    missing = [c for c in EXPECTED_COLUMNS[name] if c not in df.columns]
    if missing:
        raise SchemaDriftError(f"{name}: missing expected columns {missing}")


def load_snapshot(raw_dir: Path | str) -> Dict[str, pd.DataFrame]:
    """Read all four CSVs from ``raw_dir`` verbatim and validate each against its contract."""
    raw_dir = Path(raw_dir)
    snapshot: Dict[str, pd.DataFrame] = {}
    for name, filename in RAW_FILES.items():
        df = pd.read_csv(raw_dir / filename)
        validate_schema(name, df)
        snapshot[name] = df
    return snapshot


def _download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "octagon-dna-ingest/1.0 (educational)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed, trusted base URL
        return response.read()


def fetch_snapshot(
    dest_dir: Path | str,
    base_url: str = DEFAULT_BASE_URL,
    retries: int = 3,
    timeout: float = 60.0,
    pause_s: float = 1.0,
) -> List[Path]:
    """Download all CSVs, validate them, and only then replace the files in ``dest_dir``.

    Every file is staged and validated first, so a bad upstream release can never overwrite a good snapshot.

    Raises:
        SchemaDriftError: If a downloaded file violates the contract (after retries are exhausted for transport errors).
        OSError: If the download keeps failing at the transport level.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    staged: Dict[str, Path] = {}
    for name, filename in RAW_FILES.items():
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                payload = _download(base_url + filename, timeout)
                tmp = dest / f".{filename}.tmp"
                tmp.write_bytes(payload)
                validate_schema(name, pd.read_csv(tmp))
                staged[name] = tmp
                break
            except SchemaDriftError:
                raise
            except OSError as exc:  # network / file errors are retried politely
                last_error = exc
                time.sleep(pause_s * attempt)
        else:
            raise OSError(f"Could not fetch {filename} after {retries} attempts") from last_error
        time.sleep(pause_s)
    written: List[Path] = []
    for name, tmp in staged.items():
        final = dest / RAW_FILES[name]
        os.replace(tmp, final)
        written.append(final)
    return written
