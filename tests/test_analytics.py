"""Sandbox, anomaly detection, data quality, forecasting, charts, dashboard, report."""

import numpy as np
import pandas as pd
import pytest

from analyst.anomaly import detect_anomalies
from analyst.charts import build_chart
from analyst.dashboard import build_dashboard
from analyst.forecast import forecast_series
from analyst.quality import assess_quality
from analyst.sandbox import UnsafeCodeError, run_pandas


# ---------------------------------------------------------------- sandbox

def test_sandbox_runs_pandas(sales_df):
    out = run_pandas("result = sales.groupby('region')['revenue'].sum().idxmax()", {"sales": sales_df})
    assert out == "East"


def test_sandbox_last_expression_is_result(sales_df):
    assert run_pandas("len(dfs['sales'])", {"sales": sales_df}) == 60


@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "result = __import__('os')",
    "result = open('x.txt').read()",
    "result = sales.__class__",
    "sales.to_csv('x.csv')",
    "result = getattr(sales, 'shape')",
    "result = pd.read_csv('secret.csv')",
])
def test_sandbox_blocks_unsafe_code(sales_df, code):
    with pytest.raises(UnsafeCodeError):
        run_pandas(code, {"sales": sales_df})


def test_sandbox_does_not_mutate_source(sales_df):
    run_pandas("sales['revenue'] = 0\nresult = 1", {"sales": sales_df})
    assert sales_df["revenue"].sum() > 0


def test_sandbox_requires_result(sales_df):
    with pytest.raises(RuntimeError, match="result"):
        run_pandas("x = 1", {"sales": sales_df})


# ---------------------------------------------------------------- anomalies

def _noisy_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "product": rng.choice(["A", "B", "C"], n),
        "amount": rng.normal(100, 10, n).round(2),
        "price": 10.0,
    })


def test_detects_injected_outlier_with_reason():
    df = _noisy_frame()
    df.loc[42, "amount"] = 5000
    report = detect_anomalies(df, "t")
    hits = [a for a in report.anomalies if a.method == "outlier" and a.row_index == 42]
    assert hits and "median" in hits[0].reason


def test_no_false_positives_on_clean_normal_data():
    report = detect_anomalies(_noisy_frame(), "t")
    assert not [a for a in report.anomalies if a.method == "outlier"]


def test_detects_negative_value_and_duplicates():
    df = _noisy_frame()
    df.loc[10, "amount"] = -50
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    methods = {a.method for a in detect_anomalies(df, "t").anomalies}
    assert {"rule", "duplicate"} <= methods


def test_detects_contextual_price_error():
    df = _noisy_frame()
    df.loc[7, "price"] = 1.0
    df["price2"] = df["price"] + np.arange(len(df)) * 0  # keep numeric col set stable
    report = detect_anomalies(df, "t", columns=["price", "amount"])
    assert any(a.method == "contextual" and a.row_index == 7 for a in report.anomalies)


def test_detects_timeseries_spike():
    df = _noisy_frame()
    spike = df.iloc[:40].copy()
    spike["date"] = pd.Timestamp("2024-06-01")
    df = pd.concat([df, spike], ignore_index=True)
    report = detect_anomalies(df, "t")
    assert any(a.method == "timeseries" and a.period == "2024-06-01" for a in report.anomalies)


def test_sample_data_injected_anomalies_found(sample_store):
    report = detect_anomalies(sample_store.get("sales"), "sales")
    methods = {a.method for a in report.anomalies}
    assert {"outlier", "rule", "duplicate", "contextual", "timeseries"} <= methods
    assert len([a for a in report.anomalies if a.method == "outlier"]) <= 10  # precise, not noisy


# ---------------------------------------------------------------- quality

def test_quality_flags_issues():
    df = pd.DataFrame({"region": ["North", "north", None, "South"], "v": [1, 1, 2, 3]})
    df = pd.concat([df, df.iloc[[0]]])
    rep = assess_quality(df, "t")
    text = " ".join(rep.issues)
    assert "missing" in text and "capitalisation" in text and "duplicate" in text
    assert rep.score < 100


def test_quality_clean_data_scores_100(sales_df):
    assert assess_quality(sales_df.drop_duplicates(), "t").score == 100


# ---------------------------------------------------------------- forecast / charts / dashboard

def test_forecast_follows_trend():
    df = pd.DataFrame({"d": pd.date_range("2023-01-01", periods=730, freq="D")})
    df["v"] = np.arange(730, dtype=float)
    fc = forecast_series(df, "d", "v", periods=3)
    future = fc.frame[fc.frame["kind"] == "forecast"]["value"].to_numpy()
    assert len(future) == 3 and np.all(np.diff(future) > 0)


def test_forecast_validates_columns(sales_df):
    with pytest.raises(ValueError):
        forecast_series(sales_df, "region", "revenue")


@pytest.mark.parametrize("kind", ["bar", "line", "pie", "scatter", "area", "histogram", "box"])
def test_build_chart_types(sales_df, kind):
    fig = build_chart(sales_df, kind, x="region", y="revenue", title="t")
    assert fig.layout.title.text == "t"


def test_build_chart_rejects_bad_column(sales_df):
    with pytest.raises(ValueError, match="not in result columns"):
        build_chart(sales_df, "bar", x="nope", y="revenue")


def test_dashboard(sales_df):
    dash = build_dashboard(sales_df)
    assert ("Rows", "60") in dash.kpis and len(dash.charts) >= 2
