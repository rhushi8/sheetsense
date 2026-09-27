"""Trust boundary: model-written code is validated here before it runs."""

import ast
import warnings

import numpy as np
import pandas as pd

DENY_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "help", "memoryview", "id", "object",
}

# Writes to disk or shells out. read_* is blocked by prefix below.
DENY_ATTRS = {
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet",
    "to_feather", "to_hdf", "to_clipboard", "to_stata", "to_gbq",
    "eval", "query", "system", "popen", "save", "savez", "tofile", "load",
}

# Denylist, not allowlist: pandas lazily imports a long tail of helpers.
DENY_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "importlib",
    "builtins", "ctypes", "pickle", "requests", "urllib", "http", "glob",
    "tempfile", "webbrowser", "multiprocessing", "threading",
}


def _guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in DENY_IMPORTS:
        raise UnsafeCode(f"module not allowed: {name}")
    return __import__(name, *args, **kwargs)


# __import__ stays for pandas' C-level lazy imports. validate() still blocks underscore names.
SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "round": round, "sorted": sorted, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "str": str, "int": int, "float": float,
    "bool": bool, "range": range, "enumerate": enumerate, "zip": zip,
    "any": any, "all": all, "divmod": divmod, "print": lambda *a, **k: None,
    "__import__": _guarded_import,
}


class UnsafeCode(ValueError):
    pass


def validate(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"not valid python: {e.msg}") from e

    assigns_result = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCode("imports are not allowed")

        # No while loops, the easy way to hang.
        # ponytail: `for` over a huge range can still spin. Add a wall-clock kill if needed.
        if isinstance(node, ast.While):
            raise UnsafeCode("while loops are not allowed")

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr.startswith("read_"):
                raise UnsafeCode(f"attribute not allowed: {node.attr}")
            if node.attr in DENY_ATTRS:
                raise UnsafeCode(f"attribute not allowed: {node.attr}")

        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise UnsafeCode(f"name not allowed: {node.id}")
            if node.id in DENY_CALLS:
                raise UnsafeCode(f"name not allowed: {node.id}")
            if isinstance(node.ctx, ast.Store) and node.id == "result":
                assigns_result = True

    if not assigns_result:
        raise UnsafeCode("code must assign the answer to a variable named `result`")

    return True


def run(code, df):
    """Runs on a copy so bad code can't mutate the caller's frame."""
    validate(code)

    scope = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df.copy(),
    }
    # Models write "\d+" for r"\d+". The SyntaxWarning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(code, scope)  # noqa: S102 - validated above, which is this module's job
    return scope["result"]


def demo():
    df = pd.DataFrame({"city": ["Mumbai", "Pune", "Mumbai"], "amt": [10.0, 5.0, 7.0]})

    attacks = [
        "import os\nresult = 1",
        "result = open('secret.txt').read()",
        "result = __import__('os').listdir('.')",
        "result = df.__class__.__mro__",
        "result = df.to_csv('leak.csv')",
        "result = pd.read_csv('/etc/passwd')",
        "while True:\n    pass\nresult = 1",
        "result = eval('1+1')",
        "df.sum()",
    ]
    for bad in attacks:
        try:
            run(bad, df)
        except UnsafeCode:
            pass
        else:
            raise AssertionError(f"sandbox let this through:\n{bad}")

    assert run("result = df['amt'].sum()", df) == 22.0
    assert run("result = df.groupby('city')['amt'].sum().idxmax()", df) == "Mumbai"
    assert run("result = round(df['amt'].mean(), 2)", df) == 7.33

    # Regression: strftime needs __import__ in builtins.
    assert run("result = pd.Timestamp('2026-01-05').strftime('%Y-%m-%d')", df) == "2026-01-05"

    try:
        _guarded_import("os")
    except UnsafeCode:
        pass
    else:
        raise AssertionError("guarded import let `os` through")

    run("df['amt'] = 0\nresult = df['amt'].sum()", df)
    assert df["amt"].sum() == 22.0

    print(f"sandbox ok: {len(attacks)} attacks refused, 4 valid snippets ran")


if __name__ == "__main__":
    demo()
