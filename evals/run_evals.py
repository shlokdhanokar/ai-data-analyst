"""Evaluation framework: runs the live agent on benchmark questions and scores the answers.

Ground truth is computed directly from the sample data with pandas, so the expected answers
stay correct if the sample data is regenerated. Each case checks:
  - correctness: expected key facts appear in the answer
  - tool use:    the expected tool(s) were called
  - artifacts:   a chart / SQL was produced when required
Results (pass rate, latency, tokens) are printed and written to evals/results.json.

Usage:  python evals/run_evals.py [--only 1,3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.agent import AgentResponse, DataAnalystAgent  # noqa: E402
from analyst.config import get_settings  # noqa: E402
from analyst.data_loader import DataStore, load_csv  # noqa: E402
from analyst.llm import OpenAICompatibleClient  # noqa: E402
from analyst.logging_setup import setup_logging  # noqa: E402


@dataclass
class Case:
    question: str
    expect: Callable[[dict[str, pd.DataFrame]], list[str]]  # strings that must appear (case-insensitive)
    tools: set[str] = field(default_factory=set)            # at least one of these must be called
    artifact: str | None = None                              # artifact kind that must be produced
    follow_up_of: int | None = None                          # index of a case whose context this depends on


def _top_region(d):
    return [d["sales"].groupby("region")["revenue"].sum().idxmax()]


def _top_customers(d):
    top = d["sales"].groupby("customer_id")["revenue"].sum().nlargest(3).index
    names = d["customers"].set_index("customer_id").loc[top, "customer_name"]
    return [f"{cid}|{name}" for cid, name in zip(top, names)]  # either id or name counts


def _underperformers(d):
    s = d["sales"]
    s = s.assign(year=s["order_date"].dt.year)
    by = s.pivot_table(index="product", columns="year", values="revenue", aggfunc="sum")
    growth = (by[by.columns[-1]] / by[by.columns[0]] - 1).sort_values()
    return ["|".join(growth.index[:2])]  # either of the two fastest-declining products


def _top_segment(d):
    m = d["sales"].merge(d["customers"], on="customer_id")
    return [m.groupby("segment")["revenue"].sum().idxmax()]


CASES = [
    Case("Which region generated the highest revenue?", _top_region, {"run_sql", "run_pandas", "create_chart"}),
    Case("Show monthly sales trends.", lambda d: [], {"create_chart"}, artifact="chart"),
    Case("Which products are underperforming?", _underperformers, {"run_sql", "run_pandas", "create_chart"}),
    Case("What are the top five customers by revenue? Include their names.", _top_customers, {"run_sql", "run_pandas"}),
    Case("Generate SQL for this analysis.", lambda d: ["select"], set(), follow_up_of=3),
    Case("Detect anomalies in the sales dataset.", lambda d: ["2025-07-15"], {"detect_anomalies"}, artifact="anomalies"),
    Case("Which customer segment brings the most revenue?", _top_segment, {"run_sql", "run_pandas"}),
    Case("Forecast revenue for the next 3 months.", lambda d: [], {"forecast"}, artifact="chart"),
    Case("What is the weather in London today?", lambda d: ["not|n't|no data|unable"], set()),  # should decline gracefully
]


def _norm(text: str) -> str:
    """Normalise typographic hyphens/apostrophes the model likes to emit."""
    for ch in "‐‑‒–—−":
        text = text.replace(ch, "-")
    for ch in "     ":
        text = text.replace(ch, " ")
    return text.replace("’", "'").lower()


def score(case: Case, r: AgentResponse, data: dict[str, pd.DataFrame]) -> dict:
    text = _norm(r.answer) + " " + " ".join(str(a.data).lower() for a in r.artifacts if a.kind in ("sql", "code"))
    text = _norm(text)
    expected = case.expect(data)
    facts_ok = all(any(alt.lower() in text for alt in e.split("|")) for e in expected)
    used = {s.tool for s in r.steps}
    tools_ok = not case.tools or bool(used & case.tools)
    artifact_ok = case.artifact is None or any(a.kind == case.artifact for a in r.artifacts)
    return {
        "question": case.question,
        "passed": bool(facts_ok and tools_ok and artifact_ok and not r.error),
        "facts_ok": facts_ok, "tools_ok": tools_ok, "artifact_ok": artifact_ok,
        "expected": expected, "tools_used": sorted(used),
        "latency_s": r.latency_s, "tokens": r.usage.get("total_tokens", 0),
        "error": r.error, "answer": r.answer[:3000],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated 1-based case numbers")
    ap.add_argument("--pause", type=float, default=20, help="seconds between questions (rate limits)")
    args = ap.parse_args()

    settings = get_settings()
    setup_logging("WARNING", settings.log_dir)
    if not settings.llm_configured:
        sys.exit("Set GROQ_API_KEY (or LLM_API_KEY) to run evals.")
    store = DataStore()
    for f in sorted((ROOT / "sample_data").glob("*.csv")):
        store.add(load_csv(f.read_bytes(), f.name, existing=set(store.tables)))
    data = {n: t.df for n, t in store.tables.items()}
    llm = OpenAICompatibleClient(settings.api_key, settings.base_url, settings.model, settings.temperature)

    selected = set(range(len(CASES)))
    if args.only:
        selected = {int(x) - 1 for x in args.only.split(",")}
    results = []
    agent = DataAnalystAgent(store, llm, settings)
    for i, case in enumerate(CASES):
        if i not in selected:
            continue
        if case.follow_up_of is None:
            agent.reset()  # independent question: fresh context
        r = agent.ask(case.question)
        res = score(case, r, data)
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        print(f"[{mark}] {case.question}  ({res['latency_s']}s, {res['tokens']} tok, tools={res['tools_used']})")
        if not res["passed"]:
            print(f"       expected={res['expected']} facts={res['facts_ok']} tools={res['tools_ok']} "
                  f"artifact={res['artifact_ok']} error={res['error']}")
        time.sleep(args.pause)  # free-tier tokens-per-minute limits; avoids 429 back-off inflating latency

    n = len(results)
    passed = sum(r["passed"] for r in results)
    summary = {
        "model": settings.model,
        "pass_rate": round(passed / n, 3) if n else 0,
        "passed": passed, "total": n,
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 2) if n else 0,
        "total_tokens": sum(r["tokens"] for r in results),
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }
    (ROOT / "evals" / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n{passed}/{n} passed ({summary['pass_rate']:.0%}), avg latency {summary['avg_latency_s']}s, "
          f"{summary['total_tokens']} tokens -> evals/results.json")


if __name__ == "__main__":
    main()
