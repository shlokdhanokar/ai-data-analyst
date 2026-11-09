"""CSV ingestion, validation and the in-memory DataStore (pandas + DuckDB)."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .logging_setup import get_logger, log_event

logger = get_logger("data")

ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
DATE_HINTS = ("date", "time", "month", "day", "period", "timestamp")
_CURRENCY_RE = re.compile(r"^\s*[-+]?[$€£₹]?\s*[-+]?[\d,]*\.?\d+\s*%?\s*$")


class DataValidationError(ValueError):
    """Raised when an uploaded file cannot be used as a table."""


class UnsafeQueryError(ValueError):
    """Raised when a SQL statement is not a single read-only query."""


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape


# --------------------------------------------------------------------------- helpers


def sanitize_identifier(raw: str, fallback: str = "col") -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw).strip()).strip("_").lower()
    if not name:
        name = fallback
    if name[0].isdigit():
        name = f"{fallback}_{name}"
    return name


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def _decode(data: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise DataValidationError("Could not decode file; please save it as UTF-8 CSV.")


def _sniff_delimiter(text: str) -> str:
    sample = text[:20_000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _coerce_types(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Parse numeric-looking text ("$1,200", "12%") and date-like columns."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        non_null = s.dropna().astype(str)
        if non_null.empty:
            continue

        # numbers stored as text (currency symbols, thousands separators, percents)
        if non_null.str.match(_CURRENCY_RE).mean() >= 0.95:
            cleaned = non_null.str.replace(r"[$€£₹,%\s]", "", regex=True)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() >= 0.95:
                df[col] = pd.to_numeric(
                    s.astype(str).str.replace(r"[$€£₹,%\s]", "", regex=True), errors="coerce"
                )
                warnings.append(f"Column '{col}' converted from text to numbers.")
                continue

        # dates
        looks_like_date = any(h in col.lower() for h in DATE_HINTS)
        sample = non_null.head(500)
        if looks_like_date or sample.str.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.9:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.9:
                df[col] = pd.to_datetime(s, errors="coerce", format="mixed")
                bad = int(df[col].isna().sum() - s.isna().sum())
                if bad:
                    warnings.append(f"Column '{col}': {bad} value(s) could not be parsed as dates.")
    return df


# --------------------------------------------------------------------------- loading


def load_csv(data: bytes, filename: str, max_mb: int = 50, existing: set[str] | None = None) -> Table:
    """Validate and parse one CSV file into a Table. Raises DataValidationError."""
    if not filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise DataValidationError(f"'{filename}' is not a CSV file.")
    size_mb = len(data) / 1_048_576
    if size_mb > max_mb:
        raise DataValidationError(f"'{filename}' is {size_mb:.1f} MB; the limit is {max_mb} MB.")
    if not data.strip():
        raise DataValidationError(f"'{filename}' is empty.")

    text, encoding = _decode(data)
    delimiter = "\t" if filename.lower().endswith(".tsv") else _sniff_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"'{filename}' could not be parsed as CSV: {exc}") from exc

    warnings: list[str] = []
    if encoding not in ("utf-8", "utf-8-sig"):
        warnings.append(f"File decoded as {encoding}.")

    df = df.dropna(axis=1, how="all")
    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        df = df.dropna(axis=0, how="all")
        warnings.append(f"Dropped {empty_rows} empty row(s).")
    if df.empty or df.shape[1] == 0:
        raise DataValidationError(f"'{filename}' has no data rows.")

    original = [str(c) for c in df.columns]
    cleaned = _dedupe([sanitize_identifier(c, "col") for c in original])
    if any(o.startswith("Unnamed") for o in original):
        warnings.append("Some columns had no header and were auto-named.")
    if cleaned != original:
        warnings.append("Column names normalised to snake_case for SQL.")
    df.columns = cleaned
    df = _coerce_types(df, warnings).reset_index(drop=True)

    existing = existing or set()
    base = sanitize_identifier(Path(filename).stem, "table")
    name, i = base, 2
    while name in existing:
        name, i = f"{base}_{i}", i + 1

    log_event(logger, "csv_loaded", table=name, rows=len(df), cols=df.shape[1], encoding=encoding)
    return Table(name=name, df=df, source=filename, warnings=warnings)


# --------------------------------------------------------------------------- store


class DataStore:
    """Holds the session's tables and exposes them to DuckDB for read-only SQL."""

    def __init__(self) -> None:
        self.tables: dict[str, Table] = {}
        self._con = duckdb.connect(":memory:")
        # Defence in depth: SQL can only see registered DataFrames, never the filesystem/network.
        self._con.execute("SET enable_external_access = false")

    # -- table management
    def add(self, table: Table) -> None:
        self.tables[table.name] = table
        self._con.register(table.name, table.df)

    def remove(self, name: str) -> None:
        if name in self.tables:
            self._con.unregister(name)
            del self.tables[name]

    def clear(self) -> None:
        for name in list(self.tables):
            self.remove(name)

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Unknown table '{name}'. Available: {', '.join(self.tables) or 'none'}")
        return self.tables[name].df

    def __len__(self) -> int:
        return len(self.tables)

    @property
    def fingerprint(self) -> str:
        """Changes whenever the loaded data changes; used as part of the cache key."""
        h = hashlib.sha256()
        for name in sorted(self.tables):
            df = self.tables[name].df
            h.update(name.encode())
            h.update(str(df.shape).encode())
            h.update(str(int(pd.util.hash_pandas_object(df, index=False).sum())).encode())
        return h.hexdigest()[:16]

    # -- SQL
    def query(self, sql: str, max_rows: int = 10_000) -> pd.DataFrame:
        sql = sql.strip().rstrip(";")
        self._validate_sql(sql)
        try:
            df = self._con.execute(sql).fetch_df()
        except duckdb.Error as exc:
            raise ValueError(f"SQL error: {exc}") from exc
        return df.head(max_rows)

    def _validate_sql(self, sql: str) -> None:
        if not sql:
            raise UnsafeQueryError("Empty SQL query.")
        try:
            statements = self._con.extract_statements(sql)
        except duckdb.Error as exc:
            raise ValueError(f"SQL syntax error: {exc}") from exc
        if len(statements) != 1:
            raise UnsafeQueryError("Only a single SQL statement is allowed.")
        if statements[0].type != duckdb.StatementType.SELECT:
            raise UnsafeQueryError("Only read-only SELECT queries are allowed.")
        # Block file/network access functions even inside SELECT.
        if re.search(r"\b(read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|glob|httpfs|install|load|attach|copy)\s*\(", sql, re.I):
            raise UnsafeQueryError("File and network access functions are not allowed.")

    # -- schema for prompts
    def schema_summary(self, sample_values: int = 3) -> str:
        if not self.tables:
            return "No tables loaded."
        parts = []
        for t in self.tables.values():
            df = t.df
            lines = [f"TABLE {t.name}  ({len(df):,} rows, source: {t.source})"]
            for col in df.columns:
                s = df[col]
                dtype = _friendly_dtype(s)
                if dtype in ("integer", "float"):
                    detail = f"min={_fmt(s.min())}, max={_fmt(s.max())}"
                elif dtype == "datetime":
                    detail = f"{s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}" if s.notna().any() else "all null"
                else:
                    uniq = s.dropna().unique()
                    examples = ", ".join(repr(str(v))[:30] for v in uniq[:sample_values])
                    detail = f"{len(uniq)} distinct, e.g. {examples}"
                nulls = int(s.isna().sum())
                null_txt = f", {nulls} nulls" if nulls else ""
                lines.append(f"  - {col}: {dtype} ({detail}{null_txt})")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


def _friendly_dtype(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "text"


def _fmt(v: object) -> str:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)
    return f"{f:,.0f}" if abs(f) >= 100 or f.is_integer() else f"{f:,.2f}"
