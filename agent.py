import re
import time

import requests

import sandbox

# llama-3.3 was retired 2026-09-09. See console.groq.com/docs/models if this 404s.
MODEL = "openai/gpt-oss-120b"
URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_ATTEMPTS = 2

CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

PROMPT = """You are a data analyst. Answer the question by writing pandas code
over a DataFrame named `df` that is already loaded and already cleaned.

Rules:
- Assign the final answer to a variable named `result`.
- Use only `df`, `pd` and `np`. No imports, no file access, no printing, no plots.
- `result` must be the answer itself -- a number, a string, or a small Series --
  never a sentence and never a chart.
- Do NOT re-clean the data. Amounts are already numeric, dates are already
  datetimes, duplicate rows and the TOTAL footer are already removed, and
  inconsistent casing is already unified.
- Text columns keep the spelling the sheet used, listed below. Match those
  values exactly, or compare with .str.casefold() -- never a guessed spelling.
- Reply with one ```python code block and nothing else.

The data:
{profile}

Question: {question}"""

RETRY = """
Your previous attempt failed.

```python
{code}
```

It raised: {error}

Fix it and reply with the corrected code block only."""


def _call_groq(api_key, prompt, attempts=4):
    """Waits out 429s via Retry-After instead of failing the run."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            response = requests.post(URL, headers=headers, json=body, timeout=45)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            rate_limited = e.response.status_code == 429
            if last or not (rate_limited or e.response.status_code >= 500):
                raise
            # Trust Retry-After, capped so it can't hang the run.
            delay = float(e.response.headers.get("retry-after", 5)) if rate_limited else 1.5
            time.sleep(min(delay + 0.5, 65))
        except requests.RequestException:
            if last:
                raise
            time.sleep(1.5)


def extract_code(reply):
    match = CODE_BLOCK.search(reply)
    return (match.group(1) if match else reply).strip()


def ask(question, df, profile_text, api_key, max_attempts=MAX_ATTEMPTS):
    """Never raises on bad generated code. Failures are eval data."""
    started = time.perf_counter()
    prompt = PROMPT.format(profile=profile_text, question=question)
    code = error = None

    for attempt in range(1, max_attempts + 1):
        code = extract_code(_call_groq(api_key, prompt))
        try:
            result = sandbox.run(code, df)
        except Exception as e:  # generated code can raise anything
            error = f"{type(e).__name__}: {e}"
            prompt = PROMPT.format(profile=profile_text, question=question) + \
                RETRY.format(code=code, error=error)
            continue

        return {"result": result, "code": code, "attempts": attempt,
                "error": None, "ms": round((time.perf_counter() - started) * 1000)}

    return {"result": None, "code": code, "attempts": max_attempts,
            "error": error, "ms": round((time.perf_counter() - started) * 1000)}
