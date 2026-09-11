"""The eval harness. This is the part worth showing.

Twenty questions whose answers are computed from the source data rather than
typed by hand (see messy.py). Each one is something you would actually ask
about a sales register, and most carry a trap: a TOTAL row that double-counts,
duplicate invoices, rupee amounts stored as text, three spellings of one
category.

What gets measured:

  accuracy         did it get the right number
  first-try        right without the error being fed back, so the prompt's
                   quality without the retry flattering it
  self-repaired    broken code the agent fixed itself
  never ran        failed twice; the honest failures
  latency          what a user waits, per question

Run it before and after a prompt change. A number that moved is a result.
"It seems better" is not.

`--raw` is the control: same twenty questions, untouched spreadsheet. A score
on its own says nothing. The gap between the two runs is the whole argument
for cleaning in code first.

Run:  py -3.12 evals.py [n] [--raw]   (n = first n questions, for a smoke test)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import agent
import messy
import sheet

ROOT = Path(__file__).parent

REL_TOL = 0.005   # 0.5%, absorbs rounding but not a wrong filter
ABS_TOL = 0.01
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def to_number(value):
    """Pull a single number out of whatever the agent handed back."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Series) and len(value) == 1:
        return to_number(value.iloc[0])
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        try:
            return float(value.item())
        except (ValueError, TypeError):
            return None
    match = NUMBER.search(str(value).replace(",", ""))
    return float(match.group()) if match else None


def matches(got, expect, kind):
    if kind == "number":
        number = to_number(got)
        if number is None:
            return False
        return abs(number - expect) <= max(abs(expect) * REL_TOL, ABS_TOL)
    # Text is compared loosely on purpose. A model that returns the whole
    # ranked Series instead of the winning label still knew the answer.
    return str(expect).strip().casefold() in str(got).strip().casefold()


def show(value):
    if isinstance(value, float):
        return f"{value:,.2f}"
    text = " ".join(str(value).split())
    return text[:60] + "…" if len(text) > 60 else text


def main():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("paste"):
        print("No GROQ_API_KEY in .env. Get a free key at console.groq.com")
        return 1

    if not messy.SHEET.exists():
        print(f"No sample sheet yet. Run:  py -3.12 messy.py")
        return 1

    raw_mode = "--raw" in sys.argv
    limits = [a for a in sys.argv[1:] if a.isdigit()]

    golden = json.loads((ROOT / "golden.json").read_text(encoding="utf-8"))
    if limits:
        golden = golden[:int(limits[0])]

    if raw_mode:
        # The control: exactly what `pd.read_excel(path)` gives you, which is
        # what the agent would face if we left it to cope with the mess.
        df, report = pd.read_excel(messy.SHEET), []
        print(f"RAW (no cleaning): {len(df)} rows, {len(golden)} questions\n")
    else:
        df, report = sheet.load(messy.SHEET)
        print(f"{len(df)} rows cleaned, {len(report)} repairs, {len(golden)} questions\n")
    profile_text = sheet.profile(df, report)

    rows = []
    started = time.perf_counter()

    for i, item in enumerate(golden, start=1):
        run = agent.ask(item["question"], df, profile_text, api_key)
        ok = run["error"] is None and matches(run["result"], item["expect"], item["kind"])

        verdict = "PASS" if ok else "FAIL"
        detail = "" if ok else (
            f"   expected {show(item['expect'])} | got "
            f"{run['error'] or show(run['result'])}")
        retried = " (self-repaired)" if ok and run["attempts"] > 1 else ""
        print(f"  {verdict}  {i:>2}. {item['question'][:58]:<58} "
              f"{run['ms']:>5}ms{retried}{detail}")

        rows.append({**item, "ok": ok, "got": show(run["result"]),
                     "attempts": run["attempts"], "ms": run["ms"],
                     "error": run["error"], "code": run["code"]})

    passed = sum(r["ok"] for r in rows)
    first_try = sum(r["ok"] and r["attempts"] == 1 for r in rows)
    repaired = sum(r["ok"] and r["attempts"] > 1 for r in rows)
    never_ran = sum(r["error"] is not None for r in rows)
    latencies = sorted(r["ms"] for r in rows)
    median_ms = latencies[len(latencies) // 2]

    summary = {
        "questions": len(rows),
        "accuracy": round(passed / len(rows), 3),
        "first_try": round(first_try / len(rows), 3),
        "self_repaired": repaired,
        "never_ran": never_ran,
        "median_ms": median_ms,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "model": agent.MODEL,
        "cleaned": not raw_mode,
    }

    print(f"\n  accuracy       {passed}/{len(rows)}  ({summary['accuracy']:.0%})")
    print(f"  first try      {first_try}/{len(rows)}  ({summary['first_try']:.0%})")
    print(f"  self-repaired  {repaired}")
    print(f"  never ran      {never_ran}")
    print(f"  median         {median_ms}ms per question")

    out = ROOT / ("results_raw.json" if raw_mode else "results.json")
    out.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out.name}")

    if passed < len(rows):
        print("\nFailures worth reading (the trap column says what each one tests):")
        for r in rows:
            if not r["ok"]:
                print(f"  - {r['question']}\n      trap: {r['trap'] or '(none)'}"
                      f"\n      code: {' '.join(r['code'].split())[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
