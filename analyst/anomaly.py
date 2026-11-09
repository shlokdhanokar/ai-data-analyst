"""Explainable anomaly detection.

Three complementary detectors, each producing a human-readable reason:

1. Robust univariate outliers: modified z-score (median / MAD) confirmed by Tukey IQR fences.
2. Rule violations: negative values in a column that is otherwise non-negative, duplicate rows.
3. Time-series spikes/drops: per-period totals ("timeseries") and record counts ("volume")
   compared with a rolling median.
4. Multivariate outliers (optional): IsolationForest on numeric columns, explained by the
   features that deviate most from their median.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MODIFIED_Z_THRESHOLD = 3.5
IQR_K = 1.5


@dataclass
class Anomaly:
    method: str
    reason: str
    score: float
    row_index: int | None = None
    column: str | None = None
    value: object = None
    period: str | None = None


@dataclass
class AnomalyReport:
    table: str
    anomalies: list[Anomaly] = field(default_factory=list)
    columns_checked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        if not self.anomalies:
            return pd.DataFrame(columns=["method", "row_index", "period", "column", "value", "score", "reason"])
        df = pd.DataFrame([a.__dict__ for a in self.anomalies])
        return df[["method", "row_index", "period", "column", "value", "score", "reason"]]

    def summary(self, limit: int = 15) -> str:
        if not self.anomalies:
            return f"No anomalies found in '{self.table}' (checked: {', '.join(self.columns_checked)})."
        counts = pd.Series([a.method for a in self.anomalies]).value_counts().to_dict()
        lines = [
            f"{len(self.anomalies)} anomalies in '{self.table}' "
            f"(by method: {counts}; columns checked: {', '.join(self.columns_checked)})."
        ]
        # Show the strongest few from EACH detector so no anomaly type is crowded out.
        per_method = max(2, limit // max(1, len(counts)))
        for method in counts:
            group = sorted((a for a in self.anomalies if a.method == method), key=lambda x: -x.score)
            for a in group[:per_method]:
                loc = f"row {a.row_index}" if a.row_index is not None else f"period {a.period}"
                lines.append(f"- [{a.method}] {loc}: {a.reason}")
            if len(group) > per_method:
                lines.append(f"  (+{len(group) - per_method} more {method} anomalies)")
        lines.extend(self.notes)
        return "\n".join(lines)


def _is_id_like(name: str, s: pd.Series) -> bool:
    lname = name.lower()
    if lname == "id" or lname.endswith("_id"):
        return True
    return bool(pd.api.types.is_integer_dtype(s) and s.is_unique and s.is_monotonic_increasing)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.select_dtypes(include="number").columns
        if not _is_id_like(c, df[c]) and df[c].nunique(dropna=True) > 2
    ]


def _fmt(v: float) -> str:
    return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"


def _tail_outliers(z: pd.Series, min_z: float = MODIFIED_Z_THRESHOLD, min_gap: float = 0.75) -> list:
    """Indices in either tail beyond the largest 'natural break' in robust z-scores.

    Real-world measures are often long-tailed (many legitimately large orders), so a fixed
    cut-off flags hundreds of rows. Instead we sort the tail and cut at the biggest jump:
    values separated from the rest of the distribution by a clear gap are the anomalies.
    """
    flagged: list = []
    for sign in (1, -1):
        tail = (sign * z)[sign * z > min_z].sort_values(ascending=False)
        if tail.empty:
            continue
        vals = list(tail.to_numpy()) + [min_z]
        gaps = np.diff(vals) * -1
        cut = int(np.argmax(gaps))
        if gaps[cut] >= min_gap or tail.iloc[0] > 8:
            flagged += list(tail.index[: cut + 1])
    return flagged


def univariate_outliers(df: pd.DataFrame, columns: list[str], threshold: float = MODIFIED_Z_THRESHOLD) -> list[Anomaly]:
    out: list[Anomaly] = []
    for col in columns:
        s = df[col].dropna().astype(float)
        s = s[s >= 0] if (s >= 0).mean() > 0.98 else s  # negatives are handled by rule_violations
        if len(s) < 8:
            continue
        log_scale = bool(s.min() >= 0 and s.skew() > 2)
        t = np.log1p(s) if log_scale else s
        median = t.median()
        mad = (t - median).abs().median()
        if mad == 0:
            iqr = t.quantile(0.75) - t.quantile(0.25)
            if iqr == 0:
                continue
            scale = iqr / 1.349
        else:
            scale = mad / 0.6745
        z = (t - median) / scale
        typical = float(np.expm1(median)) if log_scale else float(median)
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        for idx in _tail_outliers(z, threshold):
            val = float(s[idx])
            ratio = f" ({val / typical:,.0f}x the typical value)" if typical > 0 and val > typical * 5 else ""
            scale_note = " on a log scale" if log_scale else ""
            out.append(Anomaly(
                method="outlier",
                row_index=int(idx),
                column=col,
                value=val,
                score=round(float(abs(z[idx])), 2),
                reason=(
                    f"{col}={_fmt(val)} vs median {_fmt(typical)}{ratio}: {abs(z[idx]):.1f} robust std-devs "
                    f"away{scale_note}, clearly separated from the rest (90% of values lie in "
                    f"[{_fmt(lo)}, {_fmt(hi)}])."
                ),
            ))
    return out


def contextual_outliers(df: pd.DataFrame, columns: list[str], min_group: int = 10, consistency: float = 0.95) -> list[Anomaly]:
    """Values that break an otherwise-fixed relationship, e.g. one product sold at a wrong unit price.

    For each low-cardinality text column and numeric column: if >=95% of a group share one value,
    any row in that group with a different value is flagged.
    """
    out: list[Anomaly] = []
    cats = [c for c in df.columns
            if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
            and 2 <= df[c].nunique() <= 50]
    for cat in cats:
        for col in columns:
            for group, s in df.groupby(cat)[col]:
                s = s.dropna()
                if len(s) < min_group:
                    continue
                mode = s.mode().iloc[0]
                share = float((s == mode).mean())
                if consistency <= share < 1.0:
                    for idx, val in s[s != mode].items():
                        out.append(Anomaly(
                            method="contextual",
                            row_index=int(idx),
                            column=col,
                            value=float(val),
                            score=round(share * 10, 2),
                            reason=f"{col}={_fmt(float(val))} for {cat}='{group}', but {share:.1%} of "
                                   f"'{group}' rows have {col}={_fmt(float(mode))} (likely a data-entry error).",
                        ))
    return out


def rule_violations(df: pd.DataFrame, columns: list[str]) -> list[Anomaly]:
    out: list[Anomaly] = []
    for col in columns:
        s = df[col]
        neg = s[s < 0]
        # Only suspicious if the column is overwhelmingly non-negative.
        if 0 < len(neg) <= max(1, int(0.02 * s.notna().sum())):
            for idx, val in neg.items():
                out.append(Anomaly(
                    method="rule",
                    row_index=int(idx),
                    column=col,
                    value=float(val),
                    score=5.0,
                    reason=f"{col}={_fmt(val)} is negative while {(s >= 0).mean():.2%} of values are non-negative "
                           f"(possible data-entry error or unrecorded return).",
                ))
    dup_mask = df.duplicated(keep="first")
    for idx in df.index[dup_mask][:50]:
        out.append(Anomaly(method="duplicate", row_index=int(idx), score=4.0,
                           reason="Exact duplicate of an earlier row (possible double-counting)."))
    return out


def timeseries_anomalies(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "D",
                         window: int = 15, threshold: float = 4.0) -> list[Anomaly]:
    """Spikes/drops in per-period record volume and per-period totals vs a rolling median."""
    indexed = df.dropna(subset=[date_col]).set_index(date_col)[value_col]
    measures = {
        "number of records": indexed.resample(freq).count(),
        f"total {value_col}": indexed.resample(freq).sum(),
    }
    out: list[Anomaly] = []
    for label, series in measures.items():
        if len(series) < window + 5:
            continue
        med = series.rolling(window, center=True, min_periods=window // 2).median()
        resid = series - med
        scale = resid.abs().median() / 0.6745
        if not scale:
            continue
        z = resid / scale
        for ts in _tail_outliers(z.dropna(), threshold, min_gap=1.5):
            zval = float(z[ts])
            kind = "spike" if zval > 0 else "drop"
            out.append(Anomaly(
                method="volume" if label == "number of records" else "timeseries",
                period=str(ts.date()),
                column=value_col,
                value=float(series[ts]),
                score=round(abs(zval), 2),
                reason=f"{label.capitalize()} on {ts.date()} was {_fmt(float(series[ts]))} vs a typical "
                       f"{_fmt(float(med[ts]))} ({kind}, {abs(zval):.1f} robust std-devs from the rolling median).",
            ))
    return out


def multivariate_outliers(df: pd.DataFrame, columns: list[str], contamination: float = 0.005) -> list[Anomaly]:
    if len(columns) < 2 or len(df) < 50:
        return []
    from sklearn.ensemble import IsolationForest

    X = df[columns].astype(float)
    X = X.fillna(X.median())
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    labels = model.fit_predict(X)
    scores = -model.score_samples(X)
    med = X.median()
    spread = (X - med).abs().median().replace(0, np.nan).fillna(X.std().replace(0, 1))
    out = []
    for idx in X.index[labels == -1]:
        dev = ((X.loc[idx] - med) / spread).abs().sort_values(ascending=False)
        top = ", ".join(f"{c}={_fmt(X.at[idx, c])} (median {_fmt(med[c])})" for c in dev.index[:2])
        out.append(Anomaly(
            method="multivariate",
            row_index=int(idx),
            score=round(float(scores[X.index.get_loc(idx)]), 3),
            reason=f"Unusual combination of values; most deviant: {top}.",
        ))
    return out


def detect_anomalies(df: pd.DataFrame, table: str = "data", columns: list[str] | None = None,
                     date_column: str | None = None, include_multivariate: bool = False) -> AnomalyReport:
    cols = [c for c in (columns or numeric_columns(df)) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    report = AnomalyReport(table=table, columns_checked=cols)
    if not cols:
        report.notes.append("No numeric columns available for anomaly detection.")
        return report

    report.anomalies += univariate_outliers(df, cols)
    report.anomalies += contextual_outliers(df, cols)
    report.anomalies += rule_violations(df, cols)

    dates = [date_column] if date_column else [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dates and dates[0] in df.columns:
        # Use the "main" measure: prefer revenue/sales/amount, else the first numeric column.
        preferred = [c for c in cols if any(k in c for k in ("revenue", "sales", "amount", "total"))]
        value_col = (preferred or cols)[0]
        report.anomalies += timeseries_anomalies(df, dates[0], value_col)

    if include_multivariate:
        flagged_rows = {a.row_index for a in report.anomalies}
        report.anomalies += [a for a in multivariate_outliers(df, cols) if a.row_index not in flagged_rows]
    return report
