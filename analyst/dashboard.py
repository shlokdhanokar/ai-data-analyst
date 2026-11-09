"""Automatic dashboard generation (no LLM needed): KPIs and charts inferred from column roles."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go

from .anomaly import numeric_columns
from .charts import build_chart

MEASURE_HINTS = ("revenue", "sales", "amount", "total", "profit", "price", "value", "cost")


@dataclass
class Dashboard:
    kpis: list[tuple[str, str]] = field(default_factory=list)
    charts: list[go.Figure] = field(default_factory=list)


def _pick_measure(df: pd.DataFrame) -> str | None:
    nums = numeric_columns(df)
    for hint in MEASURE_HINTS:
        for c in nums:
            if hint in c.lower():
                return c
    return nums[0] if nums else None


def _categories(df: pd.DataFrame, limit: int = 3) -> list[str]:
    cats = []
    for c in df.columns:
        s = df[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and 2 <= s.nunique() <= 50:
            if not c.lower().endswith("id"):
                cats.append(c)
    return sorted(cats, key=lambda c: df[c].nunique())[:limit]


def _fmt(v: float) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.2f}"


def build_dashboard(df: pd.DataFrame) -> Dashboard:
    dash = Dashboard()
    measure = _pick_measure(df)
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cats = _categories(df)

    dash.kpis.append(("Rows", f"{len(df):,}"))
    if measure:
        dash.kpis.append((f"Total {measure}", _fmt(float(df[measure].sum()))))
        dash.kpis.append((f"Avg {measure}", _fmt(float(df[measure].mean()))))
    if dates:
        d = df[dates[0]].dropna()
        if not d.empty:
            dash.kpis.append(("Date range", f"{d.min():%b %Y} – {d.max():%b %Y}"))

    if measure and dates:
        monthly = df.set_index(dates[0])[measure].resample("MS").sum().reset_index()
        monthly.columns = ["month", measure]
        dash.charts.append(build_chart(monthly, "line", x="month", y=measure, title=f"Monthly {measure}"))
    for i, cat in enumerate(cats[:3]):
        if measure:
            agg = df.groupby(cat, as_index=False)[measure].sum().sort_values(measure, ascending=False).head(12)
            kind = "pie" if i == 0 and agg.shape[0] <= 6 else "bar"
            dash.charts.append(build_chart(agg, kind, x=cat, y=measure, title=f"{measure} by {cat}"))
        else:
            counts = df[cat].value_counts().head(12).rename_axis(cat).reset_index(name="count")
            dash.charts.append(build_chart(counts, "bar", x=cat, y="count", title=f"Rows by {cat}"))
    if not dash.charts and measure:
        dash.charts.append(build_chart(df, "histogram", x=measure, title=f"Distribution of {measure}"))
    return dash
