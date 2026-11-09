"""Data quality profiling: completeness, uniqueness, validity, with a 0-100 score."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class QualityReport:
    table: str
    rows: int
    columns: int
    score: float
    issues: list[str] = field(default_factory=list)
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        head = f"Data quality for '{self.table}': score {self.score:.0f}/100 ({self.rows:,} rows x {self.columns} cols)."
        if not self.issues:
            return head + " No issues found."
        return head + "\nIssues:\n" + "\n".join(f"- {i}" for i in self.issues)


def assess_quality(df: pd.DataFrame, table: str = "data") -> QualityReport:
    issues: list[str] = []
    rows, cols = df.shape
    records = []
    penalty = 0.0

    for col in df.columns:
        s = df[col]
        missing = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        rec = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_pct": round(missing * 100, 2),
            "unique": nunique,
            "example": None if s.dropna().empty else str(s.dropna().iloc[0])[:40],
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            rec.update(min=s.min(), max=s.max(), mean=round(float(s.mean()), 2) if s.notna().any() else None)
        records.append(rec)

        if missing > 0:
            sev = "high" if missing > 0.2 else "low"
            issues.append(f"'{col}' is {missing:.1%} missing ({sev}).")
            penalty += min(missing * 40, 15)
        if nunique == 1 and rows > 1:
            issues.append(f"'{col}' has a single constant value (carries no information).")
            penalty += 2
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            types = s.dropna().map(type).value_counts()
            if len(types) > 1:
                issues.append(f"'{col}' mixes value types: {', '.join(t.__name__ for t in types.index)}.")
                penalty += 3
            stripped = s.dropna().astype(str)
            ws = int((stripped != stripped.str.strip()).sum())
            if ws:
                issues.append(f"'{col}' has {ws} value(s) with leading/trailing whitespace.")
                penalty += 1
            lower = stripped.str.strip().str.lower()
            if lower.nunique() < stripped.str.strip().nunique():
                issues.append(f"'{col}' has values that differ only by capitalisation (e.g. 'North' vs 'north').")
                penalty += 2
        if pd.api.types.is_numeric_dtype(s) and (s < 0).any() and (s >= 0).mean() > 0.98:
            issues.append(f"'{col}' has {int((s < 0).sum())} negative value(s) in an otherwise non-negative column.")
            penalty += 3
        if pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            future = int((s > pd.Timestamp.now()).sum())
            if future:
                issues.append(f"'{col}' has {future} date(s) in the future.")
                penalty += 2

    dups = int(df.duplicated().sum())
    if dups:
        issues.append(f"{dups} exact duplicate row(s).")
        penalty += min(dups / max(rows, 1) * 100, 15)

    score = max(0.0, 100.0 - penalty)
    return QualityReport(table=table, rows=rows, columns=cols, score=score, issues=issues,
                         profile=pd.DataFrame(records))
