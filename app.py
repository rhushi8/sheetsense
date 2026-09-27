import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import agent
import messy
import sheet

ROOT = Path(__file__).parent

EXAMPLES = [
    "What is the total revenue across all invoices?",
    "Which category produced the most revenue?",
    "How much revenue came from Software sold in Bengaluru?",
    "Which calendar month had the highest revenue? Answer as YYYY-MM.",
    "How much revenue has actually been collected, counting paid invoices only?",
]

# Explicit path, streamlit often starts elsewhere.
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="SheetSense", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")


def pretty(value):
    if isinstance(value, tuple):
        return ", ".join(pretty(v) for v in value)
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def scoreboard():
    runs = [("cleaned", ROOT / "results.json"), ("raw sheet", ROOT / "results_raw.json")]
    available = [(label, json.loads(p.read_text(encoding="utf-8"))["summary"])
                 for label, p in runs if p.exists()]
    if not available:
        return

    st.sidebar.subheader("Measured, not asserted")
    for label, summary in available:
        st.sidebar.metric(
            f"{label}, {summary['questions']} questions",
            f"{summary['accuracy']:.0%}",
            f"{summary['median_ms']}ms median",
            delta_color="off",
        )
    st.sidebar.caption(
        "Same twenty questions, same model. The only difference is whether the "
        "spreadsheet was cleaned in code first. Run `python evals.py` to reproduce."
    )


def main():
    st.title("SheetSense")
    st.caption("Ask a messy spreadsheet a question in plain English.")
    scoreboard()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("paste"):
        st.error("No GROQ_API_KEY found. Copy .env.example to .env and add a free "
                 "key from console.groq.com")
        st.stop()

    uploaded = st.file_uploader("Spreadsheet", type=["xlsx", "xls", "csv"])
    if uploaded is None:
        if not messy.SHEET.exists():
            st.warning("No sample sheet yet. Run `python messy.py` first.")
            st.stop()
        st.caption("Using the bundled sample: a sales register with junk header rows, "
                   "rupee amounts stored as text, three spellings of each category, "
                   "double-entered invoices and a TOTAL row at the bottom.")

    source = uploaded if uploaded is not None else messy.SHEET
    try:
        df, report = sheet.load(source)
    except Exception as e:
        st.error(f"Could not read that file: {type(e).__name__}: {e}")
        st.stop()

    left, right = st.columns([3, 2])

    with right:
        st.subheader("Fixed before the model saw it")
        for item in report:
            st.markdown(f"- {item}")
        if not report:
            st.markdown("_Nothing to fix, this sheet was already tidy._")
        with st.expander(f"Cleaned data ({len(df)} rows)"):
            st.dataframe(df.head(50), width="stretch")

    with left:
        example = st.selectbox("Try one of these", [""] + EXAMPLES)
        question = st.text_input("Or ask your own", value=example,
                                 placeholder="Which city had the highest revenue?")

        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Writing pandas…"):
                run = agent.ask(question, df, sheet.profile(df, report), api_key)

            if run["error"]:
                st.error(f"Couldn't answer that one: {run['error']}")
            elif isinstance(run["result"], (pd.DataFrame, pd.Series)):
                st.dataframe(run["result"], width="stretch")
            else:
                st.success(f"### {pretty(run['result'])}")

            note = f"{run['ms']}ms"
            if run["attempts"] > 1:
                note += f" · retried {run['attempts'] - 1}× after a failure"
            st.caption(note)
            with st.expander("The pandas it wrote"):
                st.code(run["code"], language="python")


if __name__ == "__main__":
    main()
