"""Tools the LLM can call. Each returns text for the model plus artifacts for the UI.

Tool errors are returned to the model as text (not raised) so it can self-correct,
e.g. fix a wrong column name and retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .anomaly import detect_anomalies
from .charts import CHART_TYPES, build_chart, forecast_chart
from .data_loader import DataStore
from .forecast import forecast_series
from .logging_setup import get_logger, timed
from .quality import assess_quality
from .sandbox import run_pandas

logger = get_logger("tools")


@dataclass
class Artifact:
    kind: str  # sql | code | table | chart | anomalies | quality | forecast
    title: str
    data: Any


@dataclass
class ToolResult:
    content: str
    artifacts: list[Artifact] = field(default_factory=list)
    ok: bool = True


def df_to_text(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(0 rows)"
    shown = df.head(max_rows)
    with pd.option_context("display.max_columns", 30, "display.width", 200, "display.float_format", "{:,.2f}".format):
        text = shown.to_string(index=False)
    if len(df) > max_rows:
        text += f"\n... ({len(df) - max_rows} more rows, {len(df)} total)"
    return text


# --------------------------------------------------------------------------- specs

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run one read-only DuckDB SQL SELECT over the loaded tables (table names as given in the schema). "
                           "Use for aggregations, rankings, filters and joins across tables. Returns the result rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single DuckDB SELECT statement."},
                    "title": {"type": "string", "description": "Short title describing the result."},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pandas",
            "description": "Run Python/pandas code when SQL is awkward (statistics, reshaping, correlations). Each table is "
                           "available as a DataFrame variable of the same name (also in dict `dfs`); `pd` and `np` are "
                           "imported. No imports or file I/O. Assign the final output to `result`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Create a chart for the user. Provide a SQL query that returns exactly the data to plot, then "
                           "choose chart_type and columns from that query's output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "DuckDB SELECT returning the plot data."},
                    "chart_type": {"type": "string", "enum": list(CHART_TYPES)},
                    "x": {"type": "string", "description": "Column for the x-axis (or pie labels)."},
                    "y": {"type": "string", "description": "Numeric column for the y-axis (or pie values)."},
                    "color": {"type": "string", "description": "Optional column to split series by."},
                    "title": {"type": "string"},
                },
                "required": ["sql", "chart_type", "x", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Statistically detect anomalies in a table (robust z-score + IQR outliers, negative values, "
                           "duplicates, time-series spikes, multivariate IsolationForest). Returns each anomaly with the "
                           "reason it was flagged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"},
                                "description": "Numeric columns to check; omit for all."},
                    "multivariate": {"type": "boolean", "description": "Also run IsolationForest for unusual "
                                     "combinations of values (slower, noisier). Default false."},
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_quality_report",
            "description": "Profile a table's data quality: missing values, duplicates, inconsistent categories, invalid values.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast",
            "description": "Forecast a numeric measure over time (aggregated per period) with a trend+seasonality model "
                           "and an 80% interval. Produces a chart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "periods": {"type": "integer", "description": "How many future periods (1-24).", "default": 3},
                    "freq": {"type": "string", "enum": ["D", "W", "M", "Q"], "default": "M"},
                },
                "required": ["table", "date_column", "value_column"],
            },
        },
    },
]


# --------------------------------------------------------------------------- executor


class ToolExecutor:
    def __init__(self, store: DataStore, max_rows_to_llm: int = 30) -> None:
        self.store = store
        self.max_rows = max_rows_to_llm
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "run_sql": self.run_sql,
            "run_pandas": self.run_pandas,
            "create_chart": self.create_chart,
            "detect_anomalies": self.detect_anomalies,
            "data_quality_report": self.data_quality_report,
            "forecast": self.forecast,
        }

    def execute(self, name: str, raw_args: str | dict[str, Any] | None) -> ToolResult:
        with timed(logger, "tool_call", tool=name) as log:
            try:
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            except json.JSONDecodeError as exc:
                log["ok"] = False
                return ToolResult(f"ERROR: tool arguments were not valid JSON ({exc}). Retry with valid JSON.", ok=False)
            handler = self._handlers.get(name)
            if handler is None:
                log["ok"] = False
                return ToolResult(f"ERROR: unknown tool '{name}'. Available: {', '.join(self._handlers)}.", ok=False)
            try:
                result = handler(**args)
            except TypeError as exc:
                result = ToolResult(f"ERROR: bad arguments for {name}: {exc}", ok=False)
            except Exception as exc:  # surfaced to the model so it can fix and retry
                result = ToolResult(f"ERROR: {exc}", ok=False)
            log["ok"] = result.ok
            if not result.ok:
                log["error"] = result.content[:300]
            return result

    # -- handlers
    def run_sql(self, sql: str, title: str = "Query result") -> ToolResult:
        df = self.store.query(sql)
        return ToolResult(
            content=f"{len(df)} row(s):\n{df_to_text(df, self.max_rows)}",
            artifacts=[Artifact("sql", title, sql), Artifact("table", title, df)],
        )

    def run_pandas(self, code: str, title: str = "Pandas result") -> ToolResult:
        tables = {name: t.df for name, t in self.store.tables.items()}
        value = run_pandas(code, tables)
        artifacts = [Artifact("code", title, code)]
        if isinstance(value, pd.Series):
            value = value.reset_index() if value.index.name or not isinstance(value.index, pd.RangeIndex) else value.to_frame()
        if isinstance(value, pd.DataFrame):
            if not isinstance(value.index, pd.RangeIndex):
                value = value.reset_index()
            artifacts.append(Artifact("table", title, value))
            text = df_to_text(value, self.max_rows)
        else:
            text = str(value)[:3000]
        return ToolResult(content=text, artifacts=artifacts)

    def create_chart(self, sql: str, chart_type: str, x: str, title: str, y: str | None = None,
                     color: str | None = None) -> ToolResult:
        df = self.store.query(sql)
        fig = build_chart(df, chart_type, x=x, y=y, color=color or None, title=title)
        return ToolResult(
            content=f"Chart '{title}' ({chart_type}) created from {len(df)} row(s). Data:\n{df_to_text(df, self.max_rows)}",
            artifacts=[Artifact("chart", title, fig), Artifact("sql", title, sql), Artifact("table", title, df)],
        )

    def detect_anomalies(self, table: str, columns: list[str] | None = None, multivariate: bool = False) -> ToolResult:
        df = self.store.get(table)
        report = detect_anomalies(df, table=table, columns=columns or None, include_multivariate=bool(multivariate))
        frame = report.to_frame()
        artifacts = [Artifact("anomalies", f"Anomalies in {table}", frame)] if not frame.empty else []
        # Include the flagged rows themselves so the model can describe them.
        rows = [a.row_index for a in sorted(report.anomalies, key=lambda a: -a.score) if a.row_index is not None][:8]
        context = f"\n\nFlagged rows (top {len(rows)}):\n{df_to_text(df.loc[rows])}" if rows else ""
        return ToolResult(content=report.summary() + context, artifacts=artifacts)

    def data_quality_report(self, table: str) -> ToolResult:
        report = assess_quality(self.store.get(table), table=table)
        return ToolResult(content=report.summary(), artifacts=[Artifact("quality", f"Data quality: {table}", report)])

    def forecast(self, table: str, date_column: str, value_column: str, periods: int = 3, freq: str = "M") -> ToolResult:
        fc = forecast_series(self.store.get(table), date_column, value_column, periods=periods, freq=freq)
        fig = forecast_chart(fc.frame, value_column)
        return ToolResult(
            content=fc.summary(),
            artifacts=[Artifact("chart", f"{value_column} forecast", fig),
                       Artifact("table", "Forecast", fc.frame[fc.frame["kind"] == "forecast"])],
        )
