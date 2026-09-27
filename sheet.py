import re
import warnings

import pandas as pd

# Also strips "Rs." and "%" (GST columns like "18%").
CURRENCY_CHARS = re.compile(r"(?i)(\brs\.?|\binr\b|[₹$€£,%\s])")
TOTAL_WORDS = re.compile(r"\b(total|subtotal|grand\s+total|sum)\b", re.IGNORECASE)
NULL_TEXT = {"", "nan", "none", "null", "-", "na", "n/a"}

MAX_HEADER_SCAN = 15  # rows scanned for the header
MIN_ROW_FILL = 0.5  # sparser rows are footers or separators
FOOTER_FILL = 0.7  # only sparse rows get the TOTAL check
PARSE_THRESHOLD = 0.8  # share that must parse to convert


def _text(series):
    txt = series.astype(str).str.strip()
    return txt.mask(txt.str.lower().isin(NULL_TEXT))


def _looks_numeric(value):
    try:
        float(CURRENCY_CHARS.sub("", str(value)))
        return True
    except ValueError:
        return False


def _all_blank(series):
    return _text(series).isna().all()


def find_header_row(raw):
    """Heuristic over the first 15 rows: mostly filled, mostly words, no repeats."""
    best_row, best_score = 0, -1.0
    for i in range(min(MAX_HEADER_SCAN, len(raw))):
        cells = [c for c in _text(raw.iloc[i]).dropna()]
        if len(cells) < 2:
            continue
        filled = len(cells) / raw.shape[1]
        wordy = sum(not _looks_numeric(c) for c in cells) / len(cells)
        unique = len({c.lower() for c in cells}) / len(cells)
        score = filled * wordy * unique
        if score > best_score:
            best_row, best_score = i, score
    return best_row


def _dedupe_names(names):
    seen, out = {}, []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def _to_number(series):
    values = _text(series)
    present = values.notna().sum()
    if not present:
        return None
    parsed = pd.to_numeric(values.str.replace(CURRENCY_CHARS, "", regex=True), errors="coerce")
    return parsed if parsed.notna().sum() / present >= PARSE_THRESHOLD else None


def _to_datetime(series):
    values = _text(series)
    present = values.notna().sum()
    if not present:
        return None
    # "mixed" needs newer pandas, hence the fallback. Probe warnings are noise.
    for kwargs in ({"format": "mixed"}, {}):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(values, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if parsed.notna().sum() / present >= PARSE_THRESHOLD:
            return parsed
    return None


def _canonical(spellings):
    counts = spellings.value_counts()
    tied = counts[counts == counts.max()].index
    return min(tied, key=lambda v: (not v.istitle(), v))


def _normalize_case(series):
    values = _text(series)
    folded = values.str.casefold()
    if folded.nunique() == values.nunique():
        return values, False
    known = values.dropna()
    canonical = known.groupby(folded.loc[known.index]).agg(_canonical)
    return folded.map(canonical), True


def load(path):
    name = getattr(path, "name", str(path)).lower()
    reader = pd.read_csv if name.endswith((".csv", ".txt")) else pd.read_excel
    raw = reader(path, header=None, dtype=object)
    report = []

    header_row = find_header_row(raw)
    if header_row:
        report.append(f"skipped {header_row} preamble row(s) above the real header")

    headers = [("" if pd.isna(c) else str(c).strip()) for c in raw.iloc[header_row]]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = range(body.shape[1])

    keep = [i for i in range(body.shape[1]) if not _all_blank(body[i])]
    if len(keep) < body.shape[1]:
        report.append(f"dropped {body.shape[1] - len(keep)} empty column(s)")
    body = body[keep]
    body.columns = _dedupe_names([headers[i] or f"Column {i + 1}" for i in keep])

    # Stacked exports leave a repeat header row in the data.
    header_key = tuple(str(c).strip().casefold() for c in body.columns)
    repeated = body.apply(
        lambda r: tuple("" if pd.isna(v) else str(v).strip().casefold()
                        for v in r) == header_key, axis=1)
    if repeated.any():
        report.append(f"removed {int(repeated.sum())} repeated header row(s) "
                      "pasted into the data")
        body = body[~repeated]

    filled = body.apply(lambda c: _text(c).notna())
    fill_ratio = filled.mean(axis=1)

    blanks = int((fill_ratio == 0).sum())
    if blanks:
        report.append(f"dropped {blanks} blank row(s)")

    # TOTAL check is sparse rows only, so a customer called "Total Systems" is safe.
    footers = body[(fill_ratio > 0) & (fill_ratio < MIN_ROW_FILL)]
    sparse_total = body[(fill_ratio < FOOTER_FILL) & body.apply(
        lambda r: bool(TOTAL_WORDS.search(" ".join(_text(r).dropna()))), axis=1)]
    drop_idx = fill_ratio[fill_ratio == 0].index.union(footers.index).union(sparse_total.index)
    if len(sparse_total):
        report.append(f"removed {len(sparse_total)} TOTAL/summary row(s), "
                      "which double-count every figure in the sheet")
    if len(footers.index.difference(sparse_total.index)):
        report.append(f"removed {len(footers.index.difference(sparse_total.index))} "
                      "footer/note row(s)")

    df = body.drop(index=drop_idx).reset_index(drop=True)

    numbered, dated, cased = [], [], []
    for col in df.columns:
        # Numbers first, or plain ints parse as 1970 epoch dates.
        as_number = _to_number(df[col])
        if as_number is not None:
            df[col] = as_number
            numbered.append(col)
            continue
        as_date = _to_datetime(df[col])
        if as_date is not None:
            df[col] = as_date
            dated.append(col)
            continue
        cleaned, changed = _normalize_case(df[col])
        df[col] = cleaned
        if changed:
            cased.append(col)

    if numbered:
        report.append(f"parsed to numbers: {', '.join(numbered)}")
    if dated:
        report.append(f"parsed to dates: {', '.join(dated)}")
    if cased:
        report.append(f"unified inconsistent casing in: {', '.join(cased)}")

    dupes = int(df.duplicated().sum())
    if dupes:
        df = df.drop_duplicates().reset_index(drop=True)
        report.append(f"removed {dupes} duplicate row(s)")

    return df, report


def profile(df, report):
    """Columns and types only, never rows. Keeps client data out of prompts."""
    lines = [f"{len(df)} rows x {len(df.columns)} columns", "", "Columns:"]

    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        note = f", {nulls} missing" if nulls else ""
        if pd.api.types.is_numeric_dtype(s):
            kind = f"number (min {s.min():,.2f}, max {s.max():,.2f})"
            sample = ""
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = f"date ({s.min():%Y-%m-%d} to {s.max():%Y-%m-%d})"
            sample = ""
        else:
            values = [str(v) for v in pd.Series(s.dropna().unique())]
            if len(values) <= 12:
                kind = f"text, all {len(values)} values: " + ", ".join(map(repr, sorted(values)))
                sample = ""
            else:
                kind = f"text ({s.nunique()} distinct)"
                sample = " e.g. " + ", ".join(map(repr, values[:3]))
        lines.append(f"- {col}: {kind}{note}{sample}")

    if report:
        lines += ["", "Already cleaned before you see it:"]
        lines += [f"- {item}" for item in report]
    return "\n".join(lines)


def demo():
    from messy import N_MISSING_QTY, N_ROWS, SHEET

    df, report = load(SHEET)
    print("\n".join(f"  {r}" for r in report))

    assert len(df) == N_ROWS, f"expected {N_ROWS} rows after cleaning, got {len(df)}"
    assert list(df.columns) == ["Date", "Invoice No", "Customer", "City", "Category",
                                "Qty", "Unit Price", "Amount", "Status", "Sales Rep"], df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert pd.api.types.is_numeric_dtype(df["Amount"])
    assert df["Amount"].max() < df["Amount"].sum() / 2, "the TOTAL row survived into the data"
    assert {c.casefold() for c in df["Category"]} == {"hardware", "software",
                                                     "services", "consumables"}
    assert {s.casefold() for s in df["Status"]} == {"paid", "pending", "unpaid"}
    assert int(df["Qty"].isna().sum()) == N_MISSING_QTY
    assert not df.duplicated().any()

    print(f"\ncleaner ok: {len(df)} rows, {len(df.columns)} columns, {len(report)} repairs")


if __name__ == "__main__":
    demo()
