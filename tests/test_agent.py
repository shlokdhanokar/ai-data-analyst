"""Agent loop behaviour with a scripted fake LLM (no network)."""

from analyst.agent import DataAnalystAgent
from analyst.llm import LLMError, ToolFormatError
from analyst.report import to_html, to_markdown
from analyst.tools import ToolExecutor
from tests.conftest import FakeLLM, answer, tool_call


def make_agent(store, settings, script):
    llm = FakeLLM(script)
    return DataAnalystAgent(store, llm, settings), llm


def test_tool_call_then_answer(store, settings):
    agent, llm = make_agent(store, settings, [
        tool_call("run_sql", {"sql": "SELECT region, SUM(revenue) r FROM sales GROUP BY 1 ORDER BY 2 DESC"}),
        answer("**East** has the highest revenue."),
    ])
    r = agent.ask("Which region has the highest revenue?")
    assert "East" in r.answer
    assert [s.tool for s in r.steps] == ["run_sql"] and r.steps[0].ok
    assert {a.kind for a in r.artifacts} == {"sql", "table"}
    # Tool result was fed back to the model.
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "East" in tool_msgs[0]["content"]
    assert r.usage["total_tokens"] == 15


def test_schema_in_system_prompt(store, settings):
    agent, llm = make_agent(store, settings, [answer("ok")])
    agent.ask("hi")
    assert "TABLE sales" in llm.calls[0][0]["content"]


def test_conversation_memory_includes_previous_sql(store, settings):
    agent, llm = make_agent(store, settings, [
        tool_call("run_sql", {"sql": "SELECT SUM(revenue) FROM sales"}),
        answer("Total is 26,400."),
        answer("Here is the SQL."),
    ])
    agent.ask("What is total revenue?")
    agent.ask("Generate SQL for this analysis.")
    second = llm.calls[-1]
    history = " ".join(m["content"] for m in second if m["role"] in ("user", "assistant"))
    assert "What is total revenue?" in history and "SELECT SUM(revenue) FROM sales" in history


def test_tool_error_is_returned_for_self_correction(store, settings):
    agent, llm = make_agent(store, settings, [
        tool_call("run_sql", {"sql": "SELECT nope FROM sales"}),
        tool_call("run_sql", {"sql": "SELECT COUNT(*) n FROM sales"}, "c2"),
        answer("60 rows."),
    ])
    r = agent.ask("How many rows?")
    assert [s.ok for s in r.steps] == [False, True]
    assert "ERROR" in [m for m in llm.calls[1] if m["role"] == "tool"][0]["content"]


def test_chart_tool_produces_figure(store, settings):
    agent, _ = make_agent(store, settings, [
        tool_call("create_chart", {"sql": "SELECT region, SUM(revenue) revenue FROM sales GROUP BY 1",
                                   "chart_type": "bar", "x": "region", "y": "revenue", "title": "Revenue"}),
        answer("Chart shown."),
    ])
    r = agent.ask("Show revenue by region")
    assert any(a.kind == "chart" for a in r.artifacts)


def test_max_steps_forces_final_answer(store, settings):
    steps = settings.max_tool_steps
    script = [tool_call("run_sql", {"sql": "SELECT 1"}, f"c{i}") for i in range(steps)] + [answer("final")]
    agent, llm = make_agent(store, settings, script)
    r = agent.ask("loop forever")
    assert r.answer == "final" and len(r.steps) == steps


def test_malformed_tool_call_is_retried(store, settings):
    agent, _ = make_agent(store, settings, [ToolFormatError("bad"), answer("recovered")])
    assert agent.ask("q").answer == "recovered"


def test_llm_error_is_reported_not_raised(store, settings):
    agent, _ = make_agent(store, settings, [LLMError("rate limited")])
    r = agent.ask("q")
    assert r.error and "rate limited" in r.answer


def test_cache_hit_skips_llm(store, settings):
    agent, llm = make_agent(store, settings, [answer("first")])
    agent.ask("Same question")
    agent.reset()
    r = agent.ask("same   QUESTION")
    assert r.cached and r.answer == "first" and len(llm.calls) == 1


def test_no_data_and_empty_question(settings):
    from analyst.data_loader import DataStore
    agent, _ = make_agent(DataStore(), settings, [])
    assert agent.ask("anything").error == "no_data"
    assert agent.ask("  ").error == "empty"


def test_executor_handles_bad_json_and_unknown_tool(store):
    ex = ToolExecutor(store)
    assert not ex.execute("run_sql", "{not json").ok
    assert not ex.execute("delete_everything", "{}").ok
    assert not ex.execute("run_sql", {"wrong_arg": 1}).ok


def test_executor_all_tools(sample_store):
    ex = ToolExecutor(sample_store)
    assert ex.execute("run_pandas", {"code": "result = sales['revenue'].sum()"}).ok
    assert ex.execute("detect_anomalies", {"table": "sales"}).ok
    assert ex.execute("data_quality_report", {"table": "sales"}).ok
    fc = ex.execute("forecast", {"table": "sales", "date_column": "order_date", "value_column": "revenue"})
    assert fc.ok and any(a.kind == "chart" for a in fc.artifacts)
    join = ex.execute("run_sql", {"sql": "SELECT c.segment, SUM(s.revenue) r FROM sales s "
                                         "JOIN customers c USING (customer_id) GROUP BY 1"})
    assert join.ok and "Enterprise" in join.content


def test_report_export(store, settings):
    agent, _ = make_agent(store, settings, [
        tool_call("create_chart", {"sql": "SELECT region, SUM(revenue) revenue FROM sales GROUP BY 1",
                                   "chart_type": "pie", "x": "region", "y": "revenue", "title": "Share"}),
        answer("## Done"),
    ])
    r = agent.ask("share by region")
    md, html = to_markdown(store, [r]), to_html(store, [r])
    assert "share by region" in md and "```sql" in md
    assert "<html" in html and "plotly" in html.lower()
