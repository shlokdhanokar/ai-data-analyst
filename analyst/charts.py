"""Chart construction from a DataFrame + a small, validated chart spec."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_TYPES = ("bar", "line", "pie", "scatter", "area", "histogram", "box")
PALETTE = ["#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ef4444", "#14b8a6", "#eab308", "#64748b"]


def build_chart(df: pd.DataFrame, chart_type: str, x: str | None = None, y: str | None = None,
                color: str | None = None, title: str = "") -> go.Figure:
    chart_type = (chart_type or "bar").lower()
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unsupported chart type '{chart_type}'. Use one of: {', '.join(CHART_TYPES)}.")
    if df.empty:
        raise ValueError("No data to plot.")
    for label, col in (("x", x), ("y", y), ("color", color)):
        if col and col not in df.columns:
            raise ValueError(f"{label} column '{col}' not in result columns {list(df.columns)}.")

    cols = list(df.columns)
    x = x or cols[0]
    if y is None and chart_type not in ("histogram",):
        numeric = [c for c in cols if c != x and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError("Need a numeric y column.")
        y = numeric[0]

    common = dict(title=title, color_discrete_sequence=PALETTE, template="plotly_white")
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, color=color, **common)
    elif chart_type == "line":
        fig = px.line(df.sort_values(x), x=x, y=y, color=color, markers=len(df) <= 60, **common)
    elif chart_type == "area":
        fig = px.area(df.sort_values(x), x=x, y=y, color=color, **common)
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y, hole=0.35, **common)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, **common)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x, color=color, **common)
    else:
        fig = px.box(df, x=color, y=y or x, **common)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, legend_title_text="")
    return fig


def forecast_chart(frame: pd.DataFrame, value_column: str) -> go.Figure:
    actual = frame[frame["kind"] == "actual"]
    fc = frame[frame["kind"] == "forecast"]
    # Connect the forecast line to the last actual point.
    bridge = pd.concat([actual.tail(1), fc])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bridge["period"], y=bridge["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bridge["period"], y=bridge["lower"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(249,115,22,0.18)", name="80% interval"))
    fig.add_trace(go.Scatter(x=actual["period"], y=actual["value"], name="Actual", line=dict(color=PALETTE[0], width=2)))
    fig.add_trace(go.Scatter(x=bridge["period"], y=bridge["value"], name="Forecast",
                             line=dict(color=PALETTE[1], width=2, dash="dash")))
    fig.update_layout(title=f"{value_column} forecast", template="plotly_white", height=420,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig
