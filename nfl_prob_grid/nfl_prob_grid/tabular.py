"""I/O helpers: read local files or Google Sheets, parse timestamps, atomic file writes."""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import tempfile
import urllib.request
import warnings
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .errors import InputError, RowError


# ---------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------
def google_sheet_csv_url(url: str) -> str:
    """Convert a Google Sheets browser URL to its CSV-export URL (sheet must be link-shared
    or published).  Any other URL is returned unchanged."""
    if "docs.google.com/spreadsheets" not in url or "output=csv" in url or "format=csv" in url:
        return url
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", url)
    if not m:
        return url
    g = re.search(r"[#?&]gid=(\d+)", url)
    gid = g.group(1) if g else "0"
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"


def read_table(source: str, sheet_name: str = "") -> pd.DataFrame:
    """Read CSV / TSV / XLSX from disk, or a Google Sheets / CSV URL.  Everything is read as
    text (no silent type coercion); blanks stay as empty strings."""
    src = str(source).strip()
    try:
        if src.lower().startswith(("http://", "https://")):
            with urllib.request.urlopen(google_sheet_csv_url(src), timeout=30) as resp:
                data = resp.read()
            return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
        p = Path(src)
        if not p.exists():
            raise InputError(f"Input file not found: {p}")
        suffix = p.suffix.lower()
        if suffix in (".xlsx", ".xlsm", ".xls"):
            return pd.read_excel(p, sheet_name=sheet_name or 0, dtype=str, keep_default_na=False)
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False)
    except InputError:
        raise
    except Exception as exc:  # network / parse problems
        raise InputError(f"Could not read {src!r}: {exc}") from exc


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------
def parse_timestamp(value, tz_name: str = "UTC", fmt: str = "") -> pd.Timestamp:
    """Parse to a tz-aware UTC ``Timestamp`` truncated to whole seconds.

    Accepts ISO / common strings, datetimes, Excel/Sheets serial numbers, and epoch seconds.
    Naive values are interpreted in ``tz_name``.  Raises ``RowError('bad_timestamp')``.
    """
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError("missing timestamp")
        if isinstance(value, (pd.Timestamp, datetime, date)):
            ts = pd.Timestamp(value)
        else:
            s = str(value).strip()
            if not s:
                raise ValueError("missing timestamp")
            try:
                num = float(s)
            except ValueError:
                num = None
            if num is not None:
                if 20000 < num < 100000:       # Excel / Google Sheets serial day number
                    ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(num, unit="D")
                elif 1e9 < num < 4e9:          # epoch seconds
                    ts = pd.Timestamp(num, unit="s", tz="UTC")
                else:
                    raise ValueError(f"numeric value {s!r} is not a plausible timestamp")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts = pd.to_datetime(s, format=fmt) if fmt else pd.to_datetime(s)
        if pd.isna(ts):
            raise ValueError("unparseable timestamp")
        if ts.tzinfo is None:
            ts = ts.tz_localize(ZoneInfo(tz_name), ambiguous=True, nonexistent="shift_forward")
        return ts.tz_convert("UTC").floor("s")
    except (ValueError, TypeError, OverflowError) as exc:
        raise RowError("bad_timestamp", f"cannot parse timestamp {value!r}: {exc}") from None


def parse_week(value) -> int:
    s = str(value).strip().lower()
    s = re.sub(r"^(week|wk|w)\s*", "", s)
    try:
        f = float(s)
    except ValueError:
        raise RowError("bad_week", f"cannot parse week {value!r}") from None
    if f != int(f):
        raise RowError("bad_week", f"week {value!r} is not an integer")
    return int(f)


def parse_game_date(value) -> str:
    """Return ISO date string 'YYYY-MM-DD' or '' when blank. Raises RowError on garbage."""
    if value is None or str(value).strip() == "":
        return ""
    try:
        return pd.to_datetime(str(value).strip()).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        raise RowError("bad_date", f"cannot parse date {value!r}") from None


def iso(ts) -> str:
    """Canonical timestamp string used in every persisted file."""
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------
# Writing (atomic: write to a temp file in the same directory, fsync, rename)
# ---------------------------------------------------------------------------------------
def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    atomic_write_text(path, df.to_csv(index=kw.pop("index", False), lineterminator="\n", **kw))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
