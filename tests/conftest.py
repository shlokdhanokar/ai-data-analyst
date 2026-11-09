from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.config import get_settings  # noqa: E402
from analyst.data_loader import DataStore, load_csv  # noqa: E402
from analyst.llm import LLMResponse, ToolCall  # noqa: E402


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=12, freq="MS").repeat(5),
        "region": ["North", "South", "East", "West", "Central"] * 12,
        "product": ["A", "B", "C", "D", "E"] * 12,
        "revenue": [100.0, 80.0, 120.0, 90.0, 50.0] * 12,
        "quantity": [1, 2, 3, 2, 1] * 12,
    })


@pytest.fixture
def store(sales_df) -> DataStore:
    s = DataStore()
    s.add(load_csv(sales_df.to_csv(index=False).encode(), "sales.csv"))
    return s


@pytest.fixture
def sample_store() -> DataStore:
    s = DataStore()
    for f in sorted((ROOT / "sample_data").glob("*.csv")):
        s.add(load_csv(f.read_bytes(), f.name, existing=set(s.tables)))
    return s


class FakeLLM:
    """Replays scripted responses and records the messages it received."""

    model = "fake"

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages, tools=None, on_token=None) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if on_token and item.content:
            on_token(item.content)
        return item


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> LLMResponse:
    return LLMResponse(content="", tool_calls=[ToolCall(call_id, name, json.dumps(args))],
                       usage={"total_tokens": 10})


def answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage={"total_tokens": 5})
