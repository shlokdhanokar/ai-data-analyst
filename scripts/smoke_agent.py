"""Ask the live agent a few questions against the sample data (manual smoke test)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst.agent import DataAnalystAgent
from analyst.config import get_settings
from analyst.data_loader import DataStore, load_csv
from analyst.llm import OpenAICompatibleClient

s = get_settings()
store = DataStore()
for f in sorted(Path("sample_data").glob("*.csv")):
    store.add(load_csv(f.read_bytes(), f.name, existing=set(store.tables)))
agent = DataAnalystAgent(store, OpenAICompatibleClient(s.api_key, s.base_url, s.model, s.temperature), s)
questions = sys.argv[1:] or ["Which region generated the highest revenue?", "Generate SQL for this analysis."]
for q in questions:
    r = agent.ask(q)
    print("=" * 80, "\nQ:", q, f"({r.latency_s}s, {r.usage})")
    for st in r.steps:
        print("  tool:", st.tool, "ok" if st.ok else "FAIL", str(st.arguments)[:200])
    print(r.answer)
    print("artifacts:", [a.kind for a in r.artifacts])
