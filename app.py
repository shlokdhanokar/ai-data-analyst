"""Streamlit front-end for the AI Data Analyst."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analyst.agent import AgentResponse, DataAnalystAgent
from analyst.anomaly import detect_anomalies
from analyst.config import get_settings
from analyst.dashboard import build_dashboard
from analyst.data_loader import DataStore, DataValidationError, load_csv
from analyst.llm import LLMError, OpenAICompatibleClient
from analyst.logging_setup import setup_logging
from analyst.quality import assess_quality
from analyst.report import to_html, to_markdown

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SUGGESTIONS = [
    "Which region generated the highest revenue?",
    "Show monthly sales trends.",
    "Which products are underperforming?",
    "What are the top five customers?",
    "Generate SQL for this analysis.",
    "Detect anomalies in the dataset.",
    "Forecast revenue for the next 3 months.",
    "Revenue by customer segment (join customers).",
]

settings = get_settings()
setup_logging(settings.log_level, settings.log_dir)
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.markdown(
    """<style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {background: rgba(37,99,235,.06); border-radius: 10px; padding: .6rem .9rem;}
    .small-muted {color: #64748b; font-size: .82rem;}
    </style>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- session state


def init_state() -> None:
    ss = st.session_state
    if "store" not in ss:
        ss.store = DataStore()
        ss.responses: list[AgentResponse] = []
        ss.agent = None
        ss.llm_error = None
        ss.uploaded_names: set[str] = set()
        ss.pending_question = None
    if ss.agent is None and settings.llm_configured:
        try:
            llm = OpenAICompatibleClient(settings.api_key, settings.base_url, settings.model,
                                         settings.temperature, settings.request_timeout)
            ss.agent = DataAnalystAgent(ss.store, llm, settings)
        except LLMError as exc:
            ss.llm_error = str(exc)


def add_file(data: bytes, name: str) -> None:
    store: DataStore = st.session_state.store
    if len(store) >= settings.max_files:
        st.sidebar.error(f"Maximum of {settings.max_files} files reached.")
        return
    try:
        table = load_csv(data, name, settings.max_upload_mb, existing=set(store.tables))
    except DataValidationError as exc:
        st.sidebar.error(str(exc))
        return
    store.add(table)
    for w in table.warnings:
        st.sidebar.info(f"**{table.name}**: {w}")


def reset_conversation() -> None:
    st.session_state.responses = []
    if st.session_state.agent:
        st.session_state.agent.reset()


init_state()
store: DataStore = st.session_state.store


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Upload CSVs, ask questions in plain English.")

    files = st.file_uploader("Upload CSV files", type=["csv", "tsv", "txt"], accept_multiple_files=True)
    for f in files or []:
        if f.file_id not in st.session_state.uploaded_names:
            st.session_state.uploaded_names.add(f.file_id)
            add_file(f.getvalue(), f.name)

    if st.button("Load sample datasets", use_container_width=True, disabled=bool(len(store))):
        for path in sorted(SAMPLE_DIR.glob("*.csv")):
            add_file(path.read_bytes(), path.name)
        st.rerun()

    if len(store):
        st.subheader("Tables")
        for name, t in list(store.tables.items()):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{name}**  \n<span class='small-muted'>{t.shape[0]:,} rows · {t.shape[1]} cols</span>",
                        unsafe_allow_html=True)
            if c2.button(":material/close:", key=f"rm_{name}", help=f"Remove {name}"):
                store.remove(name)
                st.rerun()
        if st.button("Clear all data", use_container_width=True):
            store.clear()
            st.session_state.uploaded_names = set()
            reset_conversation()
            st.rerun()

    st.divider()
    if st.session_state.responses:
        st.subheader("Export")
        st.download_button("Report (HTML)", to_html(store, st.session_state.responses),
                           file_name="analysis_report.html", mime="text/html", use_container_width=True)
        st.download_button("Report (Markdown)", to_markdown(store, st.session_state.responses),
                           file_name="analysis_report.md", mime="text/markdown", use_container_width=True)
        if st.button("New conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

    if settings.llm_configured and not st.session_state.llm_error:
        st.caption(f"Model: `{settings.model}`")
        agent = st.session_state.agent
        if agent:
            st.caption(f"Cache: {len(agent.cache)} entries · {agent.cache.hits} hits")
    else:
        st.warning(st.session_state.llm_error or "No LLM API key set. Add GROQ_API_KEY to .env to enable chat. "
                   "Dashboard and data-quality tabs still work.")


# --------------------------------------------------------------------------- rendering helpers


def render_response(r: AgentResponse) -> None:
    st.markdown(r.answer)
    charts = [a for a in r.artifacts if a.kind == "chart"]
    for i, a in enumerate(charts):
        st.plotly_chart(a.data, use_container_width=True, key=f"chart_{id(r)}_{i}")

    anomalies = [a for a in r.artifacts if a.kind == "anomalies"]
    for a in anomalies:
        with st.expander(f"🚨 {a.title} ({len(a.data)})", expanded=True):
            st.dataframe(a.data, use_container_width=True, hide_index=True)

    queries = [a for a in r.artifacts if a.kind in ("sql", "code")]
    if queries:
        with st.expander("🧾 Generated SQL / pandas code"):
            seen = set()
            for a in queries:
                if a.data in seen:
                    continue
                seen.add(a.data)
                st.caption(a.title)
                st.code(a.data, language="sql" if a.kind == "sql" else "python")

    tables = [a for a in r.artifacts if a.kind == "table"]
    if tables:
        with st.expander("📋 Result data"):
            for a in tables:
                st.caption(a.title)
                st.dataframe(a.data, use_container_width=True, hide_index=True)

    for a in (x for x in r.artifacts if x.kind == "quality"):
        with st.expander(f"🩺 {a.title}: {a.data.score:.0f}/100"):
            st.dataframe(a.data.profile, use_container_width=True, hide_index=True)

    if r.steps:
        with st.expander(f"🔍 Reasoning trace ({len(r.steps)} tool call{'s' if len(r.steps) != 1 else ''})"):
            for i, s in enumerate(r.steps, 1):
                icon = "✅" if s.ok else "⚠️"
                st.markdown(f"{icon} **Step {i}: `{s.tool}`** · {s.duration_ms:.0f} ms")
                st.json(s.arguments, expanded=False)
                st.text(s.preview[:400])
    meta = [f"{r.latency_s:.1f}s"]
    if r.usage.get("total_tokens"):
        meta.append(f"{r.usage['total_tokens']:,} tokens")
    if r.cached:
        meta.append("⚡ cached")
    st.caption(" · ".join(meta))


# --------------------------------------------------------------------------- main tabs

if not len(store):
    st.header("Ask your data anything")
    st.markdown(
        "Upload one or more CSV files in the sidebar (or load the sample sales data) and ask questions like "
        "*“Which region generated the highest revenue?”* or *“Detect anomalies in the dataset.”*\n\n"
        "The analyst writes and runs **SQL / pandas** against your data, builds **charts**, flags **anomalies** "
        "with explanations, **forecasts**, and explains **how it got each answer**."
    )
    st.stop()

table_names = list(store.tables)
default_idx = max(range(len(table_names)), key=lambda i: len(store.tables[table_names[i]].df))
tab_chat, tab_dash, tab_quality, tab_data = st.tabs(["💬 Chat", "📈 Dashboard", "🩺 Data quality", "🗂 Data"])

with tab_chat:
    for r in st.session_state.responses:
        with st.chat_message("user"):
            st.markdown(r.question)
        with st.chat_message("assistant"):
            render_response(r)

    if not st.session_state.responses:
        st.markdown("**Try one of these:**")
        cols = st.columns(4)
        for i, q in enumerate(SUGGESTIONS):
            if cols[i % 4].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    question = st.chat_input("Ask a question about your data…", disabled=st.session_state.agent is None)
    question = question or st.session_state.pop("pending_question", None)
    if question and st.session_state.agent:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            status = st.status("Analysing…", expanded=False)
            placeholder = st.empty()
            buffer: list[str] = []

            def on_token(tok: str) -> None:
                buffer.append(tok)
                placeholder.markdown("".join(buffer) + "▌")

            response = st.session_state.agent.ask(question, on_token=on_token)
            status.update(label=f"Done · {len(response.steps)} tool call(s)", state="complete")
            placeholder.empty()
        st.session_state.responses.append(response)
        st.rerun()

with tab_dash:
    name = st.selectbox("Table", table_names, index=default_idx, key="dash_table")
    df = store.get(name)
    dash = build_dashboard(df)
    kcols = st.columns(len(dash.kpis))
    for col, (label, value) in zip(kcols, dash.kpis):
        col.metric(label, value)
    for i in range(0, len(dash.charts), 2):
        c = st.columns(2)
        for j, fig in enumerate(dash.charts[i:i + 2]):
            c[j].plotly_chart(fig, use_container_width=True, key=f"dash_{name}_{i + j}")

with tab_quality:
    name = st.selectbox("Table", table_names, index=default_idx, key="q_table")
    df = store.get(name)
    report = assess_quality(df, name)
    c1, c2, c3 = st.columns(3)
    c1.metric("Quality score", f"{report.score:.0f}/100")
    c2.metric("Rows", f"{report.rows:,}")
    c3.metric("Issues", len(report.issues))
    for issue in report.issues:
        st.markdown(f"- {issue}")
    st.dataframe(report.profile, use_container_width=True, hide_index=True)

    st.subheader("Anomaly scan")
    multivariate = st.toggle("Include multivariate (IsolationForest)", value=False)
    anomalies = detect_anomalies(df, name, include_multivariate=multivariate)
    frame = anomalies.to_frame()
    if frame.empty:
        st.success("No anomalies found.")
    else:
        st.caption(f"{len(frame)} anomalies · columns checked: {', '.join(anomalies.columns_checked)}")
        st.dataframe(frame.sort_values("score", ascending=False), use_container_width=True, hide_index=True)

with tab_data:
    name = st.selectbox("Table", table_names, index=default_idx, key="data_table")
    t = store.tables[name]
    st.caption(f"Source: {t.source} · {t.shape[0]:,} rows × {t.shape[1]} columns")
    st.dataframe(t.df.head(500), use_container_width=True)
    with st.expander("Schema as seen by the LLM"):
        st.code(store.schema_summary(), language="text")
