"""
Streaming dashboard (Altair):
- KPI cards (latest R², MAE, total rows, new rows)
- Interactive Altair line chart with tooltips & points
- Table of the EXACT Silver rows used in the most recent training step
- Buttons to start and stop the entire data pipeline (Kafka, consumers, producer).
- Robust CSV/Parquet reads, schema coercion, alias mapping, and clear debug
"""

import os
import subprocess
import time
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

# --------------------------- Paths & Config ---------------------------

DATA_DIR = Path("data")
METRICS_CSV = DATA_DIR / "metrics.csv"
SILVER_DIR = DATA_DIR / "silver"
LAST_BATCH_PARQUET = DATA_DIR / "last_batch.parquet"

# Expected logical fields; writer may use aliases (see ALIASES)
EXPECTED_COLS = ["batch", "r2", "mae"]

# Common aliases frequently used by trainers
ALIASES = {
    "batch": ["batch", "step", "iteration", "epoch", "batch_id"],
    "r2": ["r2", "r_squared", "r2_score"],
    "mae": ["mae", "mean_abs_error", "mean_absolute_error"],
}

st.set_page_config(page_title="Streaming Model Monitor", layout="wide")
st.title("📈 Streaming Model Training Monitor")

st.markdown("""
This dashboard provides a real-time view of a streaming machine learning pipeline.
As new data is produced, it flows through Kafka, is processed into Bronze and Silver layers,
and a model is continuously retrained. You can monitor the model's performance and inspect the data being used.
""")

with st.expander("What do the pipeline components do?"):
    st.markdown("""
    - **Producer**: Reads vehicle data from a file and streams it into Kafka, simulating a real-time data source.
    - **Kafka**: A message broker that receives data from the producer and makes it available to consumers.
    - **Bronze Consumer**: Subscribes to Kafka, consumes the raw data, and writes it to Parquet files in the `data/bronze` directory. This is the raw, unfiltered data layer.
    - **Silver Consumer**: Subscribes to Kafka, consumes the raw data, cleans and transforms it (e.g., mapping brands, handling missing values), and writes it to the `data/silver` directory. This is the analysis-ready data layer.
    - **Trainer**: Watches the Silver directory for new data. When enough new data arrives, it retrains a machine learning model and logs the performance metrics, which are displayed below.
    """)

# --------------------------- Helpers ---------------------------

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def read_csv_lenient(path: Path, retries: int = 2, sleep_s: float = 0.2):
    """
    Read CSV while it might be concurrently appended.
    Uses engine='python' and on_bad_lines='skip' to ignore malformed lines.
    Returns (df, skipped_flag)
    """
    skipped = False
    last_err = None
    for _ in range(retries + 1):
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            return df, skipped
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    # Final attempt (still lenient)
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        skipped = True
        return df, skipped
    except Exception as e:
        raise e if last_err is None else last_err

def coerce_and_align_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aligns incoming metrics to EXPECTED_COLS using ALIASES.
    Derives batch/total/new if missing, coerces numerics, sorts by batch.
    Returns (aligned_df, chosen_columns_map)
    """
    if df.empty:
        return df, {k: None for k in EXPECTED_COLS}

    # Map actual columns present
    chosen = {k: pick_col(df, ALIASES[k]) for k in ALIASES}

    # Coerce numeric where applicable
    for k in ["r2", "mae"]:
        c = chosen.get(k)
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep a slim, aligned view for downstream code
    aligned = pd.DataFrame()
    for logical in EXPECTED_COLS:
        col = chosen.get(logical)
        if col and col in df.columns:
            aligned[logical] = df[col]

    # Coerce/finalize types
    if "batch" in aligned:
        aligned["batch"] = pd.to_numeric(aligned["batch"], errors="coerce").astype("Int64")
        aligned = aligned.dropna(subset=["batch"]).sort_values("batch")

    return aligned, chosen

def read_metrics():
    if not METRICS_CSV.exists():
        return pd.DataFrame(), False, {k: None for k in EXPECTED_COLS}
    raw, skipped = read_csv_lenient(METRICS_CSV)
    aligned, chosen = coerce_and_align_metrics(raw)
    # De-duplicate by batch (keep most recent)
    if not aligned.empty and "batch" in aligned.columns:
        aligned = aligned.drop_duplicates(subset=["batch"], keep="last")
    return aligned, skipped, chosen

def read_last_batch():
    if not LAST_BATCH_PARQUET.exists():
        return None, None
    try:
        df = pd.read_parquet(LAST_BATCH_PARQUET)
        return df, None
    except Exception as e:
        return None, e

def read_full_silver_data():
    """
    Loads all parquet files from the silver directory.
    NOTE: In a large-scale system, you'd sample or use a summary,
          but for this project, loading all is fine.
    """
    silver_files = sorted(SILVER_DIR.glob("**/*.parquet"))
    if not silver_files:
        return None
    try:
        df = pd.concat([pd.read_parquet(f) for f in silver_files], ignore_index=True)
        return df
    except Exception:
        return None

# --------------------------- Live Dashboard Loop ---------------------------

placeholder = st.empty()

while True:
    with placeholder.container():
        # --------------------------- Load Data ---------------------------
        metrics, skipped_lines, chosen_cols = read_metrics()
        if skipped_lines:
            st.warning("Some malformed metric rows were skipped (likely a partial write).")

        # Debug tail (helps verify schema during development)
        with st.expander("Debug: metrics.csv (tail & columns)", expanded=False):
            st.write(METRICS_CSV)
            try:
                tail_raw = pd.read_csv(METRICS_CSV, engine="python", on_bad_lines="skip").tail(8)
                st.dataframe(tail_raw, use_container_width=True)
            except Exception as e:
                st.write(f"(could not read raw metrics tail: {e})")
            st.write("Aligned columns present:", list(metrics.columns))
            st.write("Chosen column mapping:", chosen_cols)
            st.write("Aligned tail:")
            st.dataframe(metrics.tail(8), use_container_width=True)

        # --------------------------- KPIs ---------------------------
        kpi1, kpi2, _ = st.columns([1, 1, 2]) # Keep layout balanced
        if not metrics.empty:
            latest = metrics.iloc[-1]
            kpi1.metric(
                "Latest R²",
                f"{float(latest['r2']):.3f}" if "r2" in latest and pd.notna(latest["r2"]) else "—",
                help="R-squared (R²) measures how well the model's predictions fit the actual data. A value of 1.0 is a perfect fit. Higher is better.",
            )
            kpi2.metric(
                "Latest MAE",
                f"{float(latest['mae']):.3f}" if "mae" in latest and pd.notna(latest["mae"]) else "—",
                help="Mean Absolute Error (MAE) is the average of the absolute differences between the predicted and actual values. Lower is better.",
            )
        else:
            # Fallback when metrics DataFrame is completely empty
            kpi1.metric("Latest R²", "—", help="R-squared (R²) measures how well the model's predictions fit the actual data. A value of 1.0 is a perfect fit. Higher is better.")
            kpi2.metric("Latest MAE", "—", help="Mean Absolute Error (MAE) is the average of the absolute differences between the predicted and actual values. Lower is better.")

        # --------------------------- Chart & Recent Rows ---------------------------
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Training Metrics Over Time")
            if not metrics.empty and set(["batch"]).issubset(metrics.columns) and (("r2" in metrics) or ("mae" in metrics)):
                plot_df = metrics[["batch"] + [c for c in ["r2", "mae"] if c in metrics.columns]].copy()
                plot_df["batch"] = plot_df["batch"].astype(int)
                m_long = plot_df.melt(id_vars=["batch"], var_name="metric", value_name="value")
                m_long = m_long.dropna(subset=["value"])

                chart = (
                    alt.Chart(m_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("batch:Q", title="Training Batch"),
                        y=alt.Y("value:Q", title="Metric value"),
                        color=alt.Color("metric:N", title="Metric"),
                        tooltip=["batch:Q", "metric:N", alt.Tooltip("value:Q", format=".4f")],
                    )
                    .interactive()
                    .properties(height=420)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Waiting for metrics...")

        with right:
            st.subheader("Data from Most Recent Training Batch")
            st.markdown("""
            This table shows the exact rows from the Silver data layer that were used to retrain the model in the latest batch.
            """)
            last_batch, parquet_err = read_last_batch()
            if parquet_err:
                st.warning(f"Could not read last_batch.parquet (install pyarrow or fastparquet?): {parquet_err}")
            elif last_batch is not None:
                display_cols = [c for c in ["ts","make","model","year","fuel_type","l_100km","co2_gpkm"] if c in last_batch.columns]
                st.dataframe(last_batch[display_cols] if display_cols else last_batch,
                             height=420, use_container_width=True)
            else:
                st.info("Waiting for the first training iteration...")

    time.sleep(2)
