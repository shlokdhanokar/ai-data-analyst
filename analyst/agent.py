"""The data-analyst agent: an LLM tool-calling loop grounded in the loaded tables.

Flow per question:
  1. Build messages = system prompt (with live schema) + recent conversation + question.
  2. Let the model call tools (SQL, pandas, charts, anomalies, quality, forecast) for up to
     `max_tool_steps` rounds; tool results (or errors, for self-correction) are fed back.
  3. The model writes a final answer with insights and a "How I got this" reasoning section.
  4. Artifacts (SQL, code, tables, charts) and a step trace are returned to the UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .cache import LRUCache, make_key
from .config import Settings
from .data_loader import DataStore
from .llm import LLMClient, LLMError, TokenCallback, ToolFormatError, parse_args
from .logging_setup import get_logger, log_event
from .tools import TOOL_SPECS, Artifact, ToolExecutor

logger = get_logger("agent")

SYSTEM_PROMPT = """You are a senior data analyst. You answer questions about the user's CSV data, which is \
loaded as DuckDB/pandas tables. Today is {today}.

## Data
{schema}

## Rules
- Ground every number in a tool result. Never guess or invent values, columns or tables.
- Use `run_sql` for aggregations, rankings, filters and joins (DuckDB SQL; use date_trunc('month', col) for months, \
strftime for formatting). Use `run_pandas` only when SQL is awkward.
- When the user asks to see, show, plot, visualise, compare or trend something, call `create_chart` (line for time \
trends, bar for category comparisons, pie for shares of a whole, scatter for relationships).
- For anomaly / outlier / unusual-value questions call `detect_anomalies`; for data-quality questions call \
`data_quality_report`; for forecasts / predictions call `forecast`.
- "Revenue" or "sales" means the money column (e.g. revenue/sales/amount); if none exists compute quantity * price.
- "Underperforming" means low or declining results relative to peers: compare totals AND trend (e.g. recent \
period vs earlier period) before concluding.
- If a tool returns ERROR, fix the call (check column names in the schema) and retry; don't give up after one error.
- When asked for SQL or code, return it in a fenced code block. If the user says "this analysis", reuse the \
analysis from the previous turn.
- If the question cannot be answered from the data, say so plainly and suggest what data would be needed.

## Answer format (markdown, concise, no preamble)
Start with a direct one-or-two-sentence answer with the key numbers in **bold**.
Then `### Key insights` with 2-4 bullet points of business implications (comparisons, %, concentration, trends).
Then `### How I got this` with 1-3 bullets explaining which tables/columns/method you used and any assumptions \
or caveats (e.g. data gaps, anomalies that might skew results).
Do not paste large tables; the UI already shows tool results and charts. Format money with thousands separators."""

FALLBACK_NOTE = (
    "You have used the maximum number of tool calls. Using only the results above, write the final answer now "
    "in the required format. Do not call any more tools."
)


@dataclass
class Step:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: float
    preview: str


@dataclass
class AgentResponse:
    question: str
    answer: str
    artifacts: list[Artifact] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None


class DataAnalystAgent:
    def __init__(self, store: DataStore, llm: LLMClient, settings: Settings,
                 cache: LRUCache[AgentResponse] | None = None) -> None:
        self.store = store
        self.llm = llm
        self.settings = settings
        self.tools = ToolExecutor(store, settings.max_rows_to_llm)
        self.cache = cache if cache is not None else LRUCache(settings.cache_size)
        self.history: list[dict[str, str]] = []

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        self.history.clear()

    def ask(self, question: str, on_token: TokenCallback | None = None) -> AgentResponse:
        question = question.strip()
        if not question:
            return AgentResponse(question, "Please type a question.", error="empty")
        if not len(self.store):
            return AgentResponse(question, "Please upload at least one CSV file first.", error="no_data")

        key = make_key(self.store.fingerprint, self._context_signature(), question)
        if (hit := self.cache.get(key)) is not None:
            log_event(logger, "cache_hit", question=question)
            response = AgentResponse(**{**hit.__dict__, "cached": True, "latency_s": 0.0})
            self._remember(question, response)
            return response

        start = time.perf_counter()
        try:
            response = self._run(question, on_token)
        except LLMError as exc:
            log_event(logger, "agent_error", question=question, error=str(exc))
            return AgentResponse(question, f"⚠️ {exc}", error=str(exc),
                                 latency_s=round(time.perf_counter() - start, 2))
        response.latency_s = round(time.perf_counter() - start, 2)
        log_event(logger, "question_answered", question=question, latency_s=response.latency_s,
                  steps=len(response.steps), **response.usage)
        if response.error is None:
            self.cache.set(key, response)
        self._remember(question, response)
        return response

    # ------------------------------------------------------------------ internals
    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(today=time.strftime("%Y-%m-%d"), schema=self.store.schema_summary())

    def _context_signature(self) -> str:
        return "|".join(m["content"][:200] for m in self.history[-2:])

    def _remember(self, question: str, response: AgentResponse) -> None:
        """Keep a compact memory: the answer plus the SQL/code that produced it (for follow-ups)."""
        queries = [a.data for a in response.artifacts if a.kind in ("sql", "code")]
        memory = response.answer
        if queries:
            memory += "\n\n[Queries used in this answer]\n" + "\n---\n".join(dict.fromkeys(queries))
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": memory[:4000]}]
        self.history = self.history[-2 * self.settings.history_turns:]

    def _run(self, question: str, on_token: TokenCallback | None) -> AgentResponse:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        messages += self.history[-2 * self.settings.history_turns:]
        messages.append({"role": "user", "content": question})

        artifacts: list[Artifact] = []
        steps: list[Step] = []
        usage: dict[str, int] = {}
        format_retries = 0

        for _ in range(self.settings.max_tool_steps):
            try:
                resp = self.llm.complete(messages, tools=TOOL_SPECS, on_token=on_token)
            except ToolFormatError:
                # The model emitted a malformed tool call; nudge it once or twice, then continue.
                format_retries += 1
                if format_retries > 2:
                    raise LLMError("The model repeatedly produced invalid tool calls. Try rephrasing.")
                messages.append({"role": "user", "content": "Your last tool call was malformed. Call the tool "
                                 "again using valid JSON arguments that match the tool schema."})
                continue
            _add_usage(usage, resp.usage)

            if not resp.tool_calls:
                return AgentResponse(question, resp.content.strip() or "(no answer)", artifacts, steps, usage)

            messages.append(resp.assistant_message())
            for call in resp.tool_calls:
                t0 = time.perf_counter()
                result = self.tools.execute(call.name, call.arguments)
                content = result.content[: self.settings.max_tool_output_chars]
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                artifacts.extend(result.artifacts)
                steps.append(Step(call.name, parse_args(call.arguments), result.ok,
                                  round((time.perf_counter() - t0) * 1000, 1), content[:400]))

        # Step budget exhausted: ask for a final answer without tools.
        messages.append({"role": "user", "content": FALLBACK_NOTE})
        resp = self.llm.complete(messages, tools=None, on_token=on_token)
        _add_usage(usage, resp.usage)
        return AgentResponse(question, resp.content.strip(), artifacts, steps, usage)


def _add_usage(total: dict[str, int], new: dict[str, int]) -> None:
    for k, v in (new or {}).items():
        total[k] = total.get(k, 0) + int(v or 0)
