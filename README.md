# SheetSense

Ask a messy spreadsheet a question in plain English. Get a number back, and the
pandas that produced it.

Any LLM can write pandas. The part worth building was a way to find out whether
the answers are right, so the repo ships an eval suite: the same twenty
questions, run twice on the same model, once against the raw file and once
against a file that code has cleaned first.

| | cleaned first | raw spreadsheet |
|---|---|---|
| **accuracy** | **20 / 20 (100%)** | **7 / 20 (35%)** |
| correct on first attempt | 20 | 7 |
| self-repaired after an error | 0 | 0 |
| code that never ran | 0 | 0 |
| median latency | 844 ms | 6 804 ms |

Reproduce both with `python evals.py` and `python evals.py --raw`.

## What the table says

Thirteen questions failed in the raw run and not one of them crashed. Every
single failure came back as a confident, plausible, wrong number. Total revenue
read ₹73,548,800 against a true ₹36,467,200, because the model summed the TOTAL
row along with the rows that TOTAL was totalling. The invoice count read 184
where there are 180.

Code that raises gets noticed. An answer that is quietly double gets pasted
into a slide.

## Why the cleaning is in Python and not in the prompt

Every trap in the sample file has exactly one right answer. The header sits
three rows down. Amounts are text with a rupee sign. One category is spelled
`Hardware`, `hardware` and `  HARDWARE  `. Four invoices are entered twice.
A TOTAL row at the bottom doubles the whole sheet.

So none of it is the model's job. `sheet.py` fixes all of it before the model
sees anything, and lists what it changed. The agent gets a frame that is already
correct plus a description of what is in it, and writes only the analysis.

That is what the 35% column buys. It is also why an answer takes 844 ms instead
of nearly seven seconds, since the model writes one groupby rather than a page
of defensive parsing.

Quietly deleting someone's TOTAL row would be indefensible, so every change
shows up in the UI:

```
skipped 3 preamble row(s) above the real header
dropped 1 empty column(s)
removed 1 TOTAL/summary row(s), which double-count every figure in the sheet
parsed to numbers: Qty, Unit Price, Amount
unified inconsistent casing in: Category, Status
removed 4 duplicate row(s)
```

## Running code the model wrote

`sandbox.py` is the trust boundary. Model output is untrusted text that is about
to be `exec`'d, so it gets checked first. The check walks the AST rather than
matching a regex, because `"__imp" + "ort__"` beats a regex and does not beat a
parse tree. Imports, dunder names, the `getattr` family, anything that writes to
disk and `while` loops are all refused before execution, and the frame passed in
is a copy.

`python sandbox.py` fires nine attack snippets and asserts every one is refused.

One thing worth knowing if you build something similar: replacing
`__builtins__` wholesale breaks pandas. A C-level lazy import looks up
`__import__` in the calling frame's builtins, so `Timestamp.strftime` dies with
`KeyError: '__import__'` inside a locked-down sandbox. The fix is a guarded
`__import__` that refuses `os`, `subprocess`, `socket` and friends while letting
the library's own internals through. The AST layer is what stops generated
source from reaching it.

## The eval harness

`golden.json` is not written by hand. `messy.py` builds a clean frame, computes
all twenty expected answers from it, then uglifies those same rows into the
spreadsheet. One script produces the data and the ground truth, so they cannot
drift apart. Change the generator and every expected answer moves with it.

Each question records what it is trying to catch:

```json
{
  "question": "What is the total revenue across all invoices?",
  "expect": 36467200.0,
  "kind": "number",
  "trap": "the TOTAL footer row would double this if it were not removed"
}
```

Four things get measured, and each earns its place:

- `accuracy`, did it get the right number
- `first try`, right without the error being fed back, so the prompt's quality
  without the retry flattering it
- `self-repaired`, broken code the agent fixed itself
- `never ran`, failed twice, the honest failures

Numeric answers match within 0.5%, which absorbs rounding but not a wrong
filter. Text answers match loosely on purpose, since a model that returns the
whole ranked Series instead of the winning label still knew the answer.

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add one free API key to `.env`: `GROQ_API_KEY` from console.groq.com.

## Run

```
python messy.py            regenerate the sample sheet and its ground truth
streamlit run app.py       the demo
python evals.py            the 20-question suite, cleaned
python evals.py --raw      the same suite with no cleaning, the control
python sheet.py            cleaner self-check
python sandbox.py          sandbox attack suite
```

`evals.py 3` runs only the first three questions, for a smoke test.

`data/try_me_expenses.xlsx` is a second file with a different shape of mess:
`Rs.` instead of `₹`, GST stored as `18%`, ALL-CAPS headers, a Grand Total, and
a duplicate header row pasted into the middle. Upload it in the app. Testing
against it is what found three bugs the bundled sample never triggered.

## Layout

```
messy.py      builds the sample workbook and golden.json from one clean frame
sheet.py      header hunting, currency parsing, dedup, footer removal, profiling
sandbox.py    AST validation and restricted execution of generated code
agent.py      prompt to Groq to pandas to sandbox, with one self-repair attempt
evals.py      the harness: accuracy, first-try, self-repair, latency
app.py        streamlit demo
golden.json   20 questions, answers computed rather than typed
results.json  latest cleaned run, results_raw.json the latest control run
```

## Known limits

- Header detection is a heuristic, scored over the first 15 rows on fill,
  wordiness and uniqueness. It handles the common shapes. A sheet with two
  stacked header rows needs a `header_row=` override, which doesn't exist yet.
- Merged cells aren't handled. openpyxl reports them as one value plus blanks,
  and guessing the intended fill direction is still guessing.
- Dates in the sample are messy but never ambiguous: `2026-01-05` and
  `05-Jan-2026`, never `05/01/2026`. That's deliberate. An ambiguous date has no
  ground truth, and a golden set you can't defend is worse than none. Real
  Indian registers are full of `05/01/2026`, and the honest answer there is to
  ask the user rather than guess.
- Case unification keeps the sheet's commonest spelling instead of imposing
  Title Case, so an answer can come back as `services`. Inventing a spelling
  would turn a real acronym like `IBM` into `Ibm`.
- Twenty questions over one file. Enough to catch a regression, nowhere near
  enough to claim a general accuracy rate.
