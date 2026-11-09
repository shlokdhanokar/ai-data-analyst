"""Lightweight, dependency-free forecasting: linear trend + monthly seasonality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FREQ_ALIASES = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS"}


@dataclass
class Forecast:
    frame: pd.DataFrame  # columns: period, value, kind (actual|forecast), lower, upper
    method: str
    value_column: str

    def summary(self) -> str:
        fc = self.frame[self.frame["kind"] == "forecast"]
        hist = self.frame[self.frame["kind"] == "actual"]
        lines = [f"Forecast of {self.value_column} using {self.method}. "
                 f"Last actual ({hist['period'].iloc[-1]:%Y-%m-%d}): {hist['value'].iloc[-1]:,.0f}."]
        for _, r in fc.iterrows():
            lines.append(f"- {r['period']:%Y-%m-%d}: {r['value']:,.0f} (80% interval {r['lower']:,.0f} to {r['upper']:,.0f})")
        return "\n".join(lines)


def forecast_series(df: pd.DataFrame, date_column: str, value_column: str, periods: int = 3,
                    freq: str = "M", agg: str = "sum") -> Forecast:
    if date_column not in df.columns or value_column not in df.columns:
        raise ValueError(f"Columns '{date_column}' and/or '{value_column}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise ValueError(f"'{date_column}' is not a date column.")
    rule = FREQ_ALIASES.get(freq.upper(), "MS")
    periods = max(1, min(int(periods), 24))

    series = df.dropna(subset=[date_column]).set_index(date_column)[value_column].resample(rule).agg(agg)
    # Drop a trailing partial period so it doesn't drag the trend down.
    if len(series) > 3 and series.index[-1] + pd.tseries.frequencies.to_offset(rule) > df[date_column].max() + pd.Timedelta(days=1):
        series = series.iloc[:-1]
    if len(series) < 4:
        raise ValueError("Need at least 4 periods of history to forecast.")

    t = np.arange(len(series), dtype=float)
    y = series.to_numpy(dtype=float)
    use_season = rule == "MS" and len(series) >= 24
    X = [np.ones_like(t), t]
    if use_season:
        months = series.index.month.to_numpy()
        X += [(months == m).astype(float) for m in range(2, 13)]
    X = np.column_stack(X)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_std = float(np.std(y - X @ coef, ddof=min(X.shape[1], len(y) - 1)))

    future_idx = pd.date_range(series.index[-1], periods=periods + 1, freq=rule)[1:]
    tf = np.arange(len(series), len(series) + periods, dtype=float)
    Xf = [np.ones_like(tf), tf]
    if use_season:
        fm = future_idx.month.to_numpy()
        Xf += [(fm == m).astype(float) for m in range(2, 13)]
    yf = np.column_stack(Xf) @ coef
    z = 1.2816  # 80% interval

    hist = pd.DataFrame({"period": series.index, "value": y, "kind": "actual", "lower": y, "upper": y})
    fut = pd.DataFrame({"period": future_idx, "value": yf, "kind": "forecast",
                        "lower": yf - z * resid_std, "upper": yf + z * resid_std})
    method = "linear trend + monthly seasonality (OLS)" if use_season else "linear trend (OLS)"
    return Forecast(frame=pd.concat([hist, fut], ignore_index=True), method=method, value_column=value_column)
