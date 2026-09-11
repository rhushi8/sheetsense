"""Check and run model-written pandas. This is the trust boundary.

Whatever the model returns is untrusted text that is about to be exec'd, so
this decides what may run before any of it does. Validation walks the AST
instead of matching a regex, because `"__imp" + "ort__"` beats a regex and
does not beat a parse tree.

Allowlist for names, denylist for the pandas methods that touch disk. If a
check here is wrong the cost is someone's filesystem, so this file gets the
paranoid treatment the rest of the repo doesn't need.
"""

import ast
import warnings

import numpy as np
import pandas as pd

# Callables that would get code out of the sandbox.
DENY_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "help", "memoryview", "id", "object",
}

# pandas/numpy methods that write to disk or shell out. Reading is covered by
# the `read_` prefix rule below. The frame is already loaded, so nothing here
# needs to open a file.
DENY_ATTRS = {
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet",
    "to_feather", "to_hdf", "to_clipboard", "to_stata", "to_gbq",
    "eval", "query", "system", "popen", "save", "savez", "tofile", "load",
}

# Modules that would let code leave the process. Everything else is allowed:
# pandas and numpy lazily import a long tail of helpers, and an allowlist
# turns every new pandas method into a mystery failure.
DENY_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "importlib",
    "builtins", "ctypes", "pickle", "requests", "urllib", "http", "glob",
    "tempfile", "webbrowser", "multiprocessing", "threading",
}


def _guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in DENY_IMPORTS:
        raise UnsafeCode(f"module not allowed: {name}")
    return __import__(name, *args, **kwargs)


# Deliberately small. Anything missing here is unavailable to generated code.
#
# `__import__` is the exception and it is not the hole it looks like. A C-level
# lazy import inside pandas looks up `__import__` in the calling frame's
# builtins, which is this dict, so without it Timestamp.strftime raises
# KeyError. Generated source still can't reach it: validate() rejects any name
# starting with an underscore, and getattr and friends are denied outright.
# The guard above is the second lock, for imports below the Python layer.
SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "round": round, "sorted": sorted, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "str": str, "int": int, "float": float,
    "bool": bool, "range": range, "enumerate": enumerate, "zip": zip,
    "any": any, "all": all, "divmod": divmod, "print": lambda *a, **k: None,
    "__import__": _guarded_import,
}


class UnsafeCode(ValueError):
    """Generated code failed validation and was never executed."""


def validate(code):
    """Raise UnsafeCode unless every node in the tree is permitted."""
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

        # A while loop is the one easy way for one-shot analysis to hang.
        # Comprehensions and vectorised pandas cover every real case.
        # ponytail: `for` is still allowed, so a huge literal range could spin.
        # Add a wall-clock kill if this ever runs somewhere you can't close.
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
    """Validate, then run against a copy of the frame. Returns `result`.

    The copy matters. Generated code can mutate what it is given, and the
    caller's frame has to survive a bad answer for the next question.
    """
    validate(code)

    scope = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df.copy(),
    }
    # Models write `"\d+"` where they mean `r"\d+"` all the time. It works, and
    # the SyntaxWarning is noise about someone else's style.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(code, scope)  # noqa: S102 - validated above, which is this module's job
    return scope["result"]


def demo():
    """Check the dangerous things are refused and the useful ones still run."""
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
        "df.sum()",  # never assigns `result`
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

    # Regression: strftime lazily imports at C level, which needs __import__ in
    # this frame's builtins. Locking builtins down too hard broke it.
    assert run("result = pd.Timestamp('2026-01-05').strftime('%Y-%m-%d')", df) == "2026-01-05"

    # The modules that matter are still refused, even below the Python layer.
    try:
        _guarded_import("os")
    except UnsafeCode:
        pass
    else:
        raise AssertionError("guarded import let `os` through")

    # Code that mutates its copy leaves the caller's frame alone.
    run("df['amt'] = 0\nresult = df['amt'].sum()", df)
    assert df["amt"].sum() == 22.0

    print(f"sandbox ok: {len(attacks)} attacks refused, 4 valid snippets ran")


if __name__ == "__main__":
    demo()
