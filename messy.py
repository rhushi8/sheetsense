"""Builds the messy workbook and its answers together so they can't drift."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SHEET = DATA / "sales_register.xlsx"
GOLDEN = ROOT / "golden.json"

SEED = 7
N_ROWS = 180
N_DUPES = 4
N_BLANK_ROWS = 3
N_MISSING_QTY = 6

CITIES = ["Mumbai", "Pune", "Bengaluru", "Delhi", "Ahmedabad"]
CATEGORIES = ["Hardware", "Software", "Services", "Consumables"]
REPS = ["A. Iyer", "B. Khan", "C. Dsouza", "D. Mehta"]
CUSTOMERS = [
    f"{prefix} {suffix}"
    for prefix in ("Sunrise", "Apex", "Metro", "Vertex", "Orbit", "Nimbus", "Kalpataru", "Zenith")
    for suffix in ("Enterprises", "Systems", "Traders")
]
STATUSES = ["Paid"] * 60 + ["Pending"] * 25 + ["Unpaid"] * 15

HEADER = ["Date", "Invoice No", "Customer", "City", "", "Category",
          "Qty", "Unit Price", "Amount", "Status", "Sales Rep"]
N_COLS = len(HEADER)


def build_clean(rng):
    start = pd.Timestamp("2025-07-01")
    qty = rng.integers(1, 41, N_ROWS).astype(float)
    price = rng.choice([500, 1200, 2500, 4800, 7500, 12000, 18000, 25000], N_ROWS).astype(float)

    df = pd.DataFrame({
        "Date": start + pd.to_timedelta(rng.integers(0, 365, N_ROWS), unit="D"),
        "Invoice No": [f"HT-{2000 + i}" for i in range(N_ROWS)],
        "Customer": rng.choice(CUSTOMERS, N_ROWS),
        "City": rng.choice(CITIES, N_ROWS),
        "Category": rng.choice(CATEGORIES, N_ROWS),
        "Qty": qty,
        "Unit Price": price,
        "Amount": qty * price,
        "Status": rng.choice(STATUSES, N_ROWS),
        "Sales Rep": rng.choice(REPS, N_ROWS),
    })

    # Some rows lose Qty, so Amount is the revenue source of truth.
    df.loc[rng.choice(N_ROWS, N_MISSING_QTY, replace=False), "Qty"] = np.nan
    return df


def _messy_row(row, i, rng):
    cells = [""] * N_COLS

    cells[0] = row.Date.strftime("%Y-%m-%d") if i % 2 else row.Date.strftime("%d-%b-%Y")
    cells[1] = str(row["Invoice No"])
    cells[2] = str(row["Customer"])
    cells[3] = str(row["City"])
    category = str(row["Category"])
    cells[5] = str(rng.choice([category, category.lower(), f"  {category.upper()}  "]))
    cells[6] = "" if pd.isna(row["Qty"]) else int(row["Qty"])
    cells[7] = float(row["Unit Price"])
    amount = float(row["Amount"])
    cells[8] = f"₹ {amount:,.2f}" if rng.random() < 0.7 else amount
    status = str(row["Status"])
    cells[9] = str(rng.choice([status, status.upper(), status.lower()]))
    cells[10] = str(row["Sales Rep"])
    return cells


def messify(clean, rng):
    rows = [
        ["Horizon Traders — Sales Register"] + [""] * (N_COLS - 1),
        ["Confidential — internal use only"] + [""] * (N_COLS - 1),
        [""] * N_COLS,
        list(HEADER),
    ]

    body = [_messy_row(row, i, rng) for i, (_, row) in enumerate(clean.iterrows())]

    for idx in rng.choice(len(body), N_DUPES, replace=False):
        body.insert(int(idx), list(body[int(idx)]))

    for idx in rng.choice(len(body), N_BLANK_ROWS, replace=False):
        body.insert(int(idx), [""] * N_COLS)

    rows.extend(body)

    total = ["", "", "TOTAL", "", "", "", "", "", float(clean.Amount.sum()), "", ""]
    rows.append(total)
    rows.append([""] * N_COLS)
    rows.append(["Prepared by: R. Sangle"] + [""] * (N_COLS - 1))
    rows.append(["Figures in INR"] + [""] * (N_COLS - 1))
    return pd.DataFrame(rows)


def build_golden(clean):
    by_city = clean.groupby("City").Amount.sum()
    by_cat = clean.groupby("Category").Amount.sum()
    by_rep = clean.groupby("Sales Rep").Amount.sum()
    by_month = clean.groupby(clean.Date.dt.to_period("M")).Amount.sum()
    q1 = clean[(clean.Date >= "2026-01-01") & (clean.Date <= "2026-03-31")]

    q = [
        ("What is the total revenue across all invoices?",
         clean.Amount.sum(), "number",
         "the TOTAL footer row would double this if it were not removed"),
        ("How much revenue came from Mumbai?",
         by_city["Mumbai"], "number", "requires the rupee-text column to be numeric"),
        ("How many invoices are in this register?",
         len(clean), "number", "duplicate rows inflate this by 4 if not dropped"),
        ("Which category produced the most revenue?",
         by_cat.idxmax(), "text", "mixed casing splits the groups if not normalised"),
        ("What is the average invoice amount?",
         clean.Amount.mean(), "number", "footer row skews both sum and count"),
        ("How many invoices are still unpaid?",
         int((clean.Status == "Unpaid").sum()), "number", "status casing is inconsistent"),
        ("What was the total revenue between January and March 2026 inclusive?",
         q1.Amount.sum(), "number", "dates arrive as two different text formats"),
        ("Which sales rep brought in the most revenue?",
         by_rep.idxmax(), "text", ""),
        ("How many distinct customers appear in the register?",
         clean.Customer.nunique(), "number", ""),
        ("What is the largest single invoice amount?",
         clean.Amount.max(), "number", "the TOTAL row is larger than any real invoice"),
        ("What percentage of total revenue came from the highest-earning city?",
         100 * by_city.max() / clean.Amount.sum(), "number", ""),
        ("Which calendar month had the highest revenue? Answer as YYYY-MM.",
         str(by_month.idxmax()), "text", ""),
        ("How many rows are missing a quantity value?",
         N_MISSING_QTY, "number", "blank cells, not zeros"),
        ("What is the median unit price?",
         clean["Unit Price"].median(), "number", ""),
        ("How much revenue came from Software sold in Bengaluru?",
         clean[(clean.Category == "Software") & (clean.City == "Bengaluru")].Amount.sum(),
         "number", "two-condition filter across a case-normalised column"),
        ("How many invoices are worth more than 50000?",
         int((clean.Amount > 50000).sum()), "number", ""),
        ("What is the total quantity sold across all invoices?",
         clean.Qty.sum(), "number", "missing quantities must be skipped, not zeroed"),
        ("What is the average invoice amount for the Services category?",
         clean[clean.Category == "Services"].Amount.mean(), "number", ""),
        ("What is the date of the earliest invoice? Answer as YYYY-MM-DD.",
         clean.Date.min().date().isoformat(), "text", ""),
        ("How much revenue has actually been collected, counting paid invoices only?",
         clean[clean.Status == "Paid"].Amount.sum(), "number", "status casing is inconsistent"),
    ]

    return [
        {"question": text,
         "expect": round(float(expect), 4) if kind == "number" else str(expect),
         "kind": kind,
         "trap": trap}
        for text, expect, kind, trap in q
    ]


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)

    clean = build_clean(rng)
    messify(clean, rng).to_excel(SHEET, index=False, header=False)
    golden = build_golden(clean)
    GOLDEN.write_text(json.dumps(golden, indent=2), encoding="utf-8")

    print(f"wrote {SHEET.relative_to(ROOT)} "
          f"({N_ROWS} real rows + {N_DUPES} duplicates + {N_BLANK_ROWS} blanks "
          f"+ 3 preamble + 1 TOTAL + 2 footer)")
    print(f"wrote {GOLDEN.name} ({len(golden)} questions, answers computed from the clean frame)")


if __name__ == "__main__":
    main()
