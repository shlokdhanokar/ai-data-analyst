"""Restricted execution of LLM-generated pandas code.

The code is parsed and statically checked before running: no imports, no dunder access,
no file/system builtins. It runs with a whitelisted set of builtins and only `pd`, `np`
and copies of the loaded DataFrames in scope, and must assign its output to `result`.

This is a guard-rail against accidental or prompt-injected misuse, not a hardened
security boundary. For untrusted multi-tenant deployments run it in a separate
process/container (see README, "Limitations").
"""

from __future__ import annotations

import ast
import builtins
from typing import Any

import numpy as np
import pandas as pd

FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview",
}
FORBIDDEN_ATTRS = {
    "to_csv", "to_excel", "to_parquet", "to_pickle", "to_sql", "to_json", "to_hdf", "to_feather",
    "read_csv", "read_excel", "read_parquet", "read_pickle", "read_sql", "read_json", "read_html",
    "read_hdf", "read_feather", "eval", "query_eval", "system", "popen",
}
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
        "len", "list", "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str",
        "sum", "tuple", "zip", "print", "ValueError", "KeyError", "Exception",
    )
}


class UnsafeCodeError(ValueError):
    pass


def validate_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Syntax error in pandas code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Imports are not allowed; `pd` and `np` are already available.")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCodeError(f"'{type(node).__name__}' statements are not allowed.")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_CALLS):
            raise UnsafeCodeError(f"Use of '{node.id}' is not allowed.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCodeError("Access to private/dunder attributes is not allowed.")
            if node.attr in FORBIDDEN_ATTRS:
                raise UnsafeCodeError(f"'.{node.attr}' is not allowed (no file or system I/O).")
    return tree


def run_pandas(code: str, tables: dict[str, pd.DataFrame]) -> Any:
    """Execute validated code; returns the value bound to `result`."""
    tree = validate_code(code)
    scope: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    copies = {name: df.copy() for name, df in tables.items()}
    scope["dfs"] = copies
    scope.update(copies)

    # If the last statement is a bare expression, treat it as the result.
    if tree.body and isinstance(tree.body[-1], ast.Expr) and "result" not in code:
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())], value=tree.body[-1].value
        )
        ast.fix_missing_locations(tree)

    try:
        exec(compile(tree, "<analysis>", "exec"), scope)  # noqa: S102 - validated above
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

    if "result" not in scope:
        raise RuntimeError("The code must assign its final output to a variable named `result`.")
    return scope["result"]
