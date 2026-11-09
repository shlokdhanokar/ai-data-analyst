"""Export the analysis session as a Markdown or self-contained HTML report."""

from __future__ import annotations

import html
import time
from typing import Iterable

import markdown_it
import pandas as pd

from .agent import AgentResponse
from .data_loader import DataStore

_md = markdown_it.MarkdownIt("commonmark").enable("table")


def _data_section(store: DataStore) -> str:
    lines = ["## Datasets", ""]
    for t in store.tables.values():
        lines.append(f"- **{t.name}** ({t.source}): {t.df.shape[0]:,} rows x {t.df.shape[1]} columns")
    return "\n".join(lines)


def to_markdown(store: DataStore, responses: Iterable[AgentResponse]) -> str:
    out = [f"# AI Data Analyst Report", f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_", "", _data_section(store), ""]
    for i, r in enumerate(responses, 1):
        out += [f"## Q{i}. {r.question}", "", r.answer, ""]
        for a in r.artifacts:
            if a.kind == "sql":
                out += [f"```sql\n{a.data}\n```", ""]
            elif a.kind == "code":
                out += [f"```python\n{a.data}\n```", ""]
            elif a.kind in ("table", "anomalies") and isinstance(a.data, pd.DataFrame) and not a.data.empty:
                out += [f"**{a.title}**", "", _md_table(a.data.head(20)), ""]
    return "\n".join(out)


def _md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    def cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:,.2f}"
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def to_html(store: DataStore, responses: Iterable[AgentResponse]) -> str:
    responses = list(responses)
    body = [f"<h1>AI Data Analyst Report</h1><p class='muted'>Generated {time.strftime('%Y-%m-%d %H:%M')}</p>",
            _md.render(_data_section(store))]
    first_chart = True
    for i, r in enumerate(responses, 1):
        body.append(f"<section><h2>Q{i}. {html.escape(r.question)}</h2>{_md.render(r.answer)}")
        for a in r.artifacts:
            if a.kind == "chart":
                body.append(a.data.to_html(full_html=False, include_plotlyjs="cdn" if first_chart else False))
                first_chart = False
            elif a.kind in ("sql", "code"):
                body.append(f"<pre><code>{html.escape(a.data)}</code></pre>")
            elif a.kind in ("table", "anomalies") and isinstance(a.data, pd.DataFrame) and not a.data.empty:
                body.append(f"<h4>{html.escape(a.title)}</h4>" + a.data.head(20).to_html(index=False, border=0))
        body.append("</section>")
    style = """
    body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#0f172a}
    section{border-top:1px solid #e2e8f0;padding-top:1rem;margin-top:1.5rem}
    pre{background:#f1f5f9;padding:.75rem;border-radius:6px;overflow-x:auto}
    table{border-collapse:collapse;font-size:.85rem;margin:.5rem 0}
    th,td{border-bottom:1px solid #e2e8f0;padding:.3rem .6rem;text-align:left}
    .muted{color:#64748b}"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>AI Data Analyst Report</title><style>{style}</style></head><body>{''.join(body)}</body></html>"
