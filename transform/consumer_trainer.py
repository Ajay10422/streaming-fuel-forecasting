"""
Trainer that watches the SILVER parquet directory and retrains whenever
new rows appear. Appends metrics to data/metrics.csv and writes the exact
rows used in the most recent retrain to data/last_batch.parquet (for the UI).

Metrics now include: batch, r2, mae, total_rows, new_rows, timestamp.
"""

import os
import json
import time
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

SILVER_DIR = os.path.join("data", "silver")
DATA_DIR   = "data"
METRICS_CSV = os.path.join(DATA_DIR, "metrics.csv")
STATE_JSON  = os.path.join(DATA_DIR, "metrics_state.json")
LAST_BATCH_PARQUET = os.path.join(DATA_DIR, "last_batch.parquet")

os.makedirs(DATA_DIR, exist_ok=True)

POLL_SECS = 3
MIN_ROWS_TO_TRAIN    = 20       # don't train before this many rows exist
INCREMENT_THRESHOLD  = 10       # train when at least this many new rows arrive


def _list_silver_files():
    return sorted(glob(os.path.join(SILVER_DIR, "**", "*.parquet"), recursive=True))

def _load_silver_df() -> pd.DataFrame:
    dfs = []
    for path in _list_silver_files():
        try:
            dfs.append(pd.read_parquet(path, engine="pyarrow"))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def _total_silver_rows() -> int:
    # Sum counts quickly (we just load and sum; small scale project)
    total = 0
    for path in _list_silver_files():
        try:
            total += pd.read_parquet(path, engine="pyarrow").shape[0]
        except Exception:
            pass
    return total

def _read_state() -> dict:
    if os.path.exists(STATE_JSON):
        try:
            with open(STATE_JSON, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _write_state(state: dict) -> None:
    with open(STATE_JSON, "w") as f:
        json.dump(state, f)

def _append_metrics_csv(row: dict):
    # Ensure header + flush to make Streamlit see it immediately
    file_exists = os.path.exists(METRICS_CSV)
    cols = ["batch", "r2", "mae", "timestamp"]
    line = ",".join(str(row[c]) for c in cols) + "\n"
    with open(METRICS_CSV, "a", newline="") as f:
        if not file_exists or os.path.getsize(METRICS_CSV) == 0:
            f.write(",".join(cols) + "\n")
        f.write(line)
        f.flush()
        os.fsync(f.fileno())

def main():
    state = _read_state()
    last_rows = int(state.get("last_rows", 0))
    batch_idx = int(state.get("batch_idx", 0))

    print(f"[trainer] Watching {SILVER_DIR} (poll={POLL_SECS}s). Last rows={last_rows}")

    while True:
        try:
            total_rows = _total_silver_rows()

            # Trigger retrain only when enough new rows accumulated
            new_rows = total_rows - last_rows
            if total_rows >= MIN_ROWS_TO_TRAIN and new_rows >= INCREMENT_THRESHOLD:
                df = _load_silver_df()
                if df.empty:
                    time.sleep(POLL_SECS); continue

                # Ensure columns & basic clean (Silver should already be mostly clean)
                required = {"year","l_100km","co2_gpkm"}
                if not required.issubset(df.columns):
                    print(f"[trainer] Missing required columns in Silver: {required - set(df.columns)}")
                    time.sleep(POLL_SECS); continue

                # Pick the exact NEW rows used in this retrain (sorted by ts if present)
                if "ts" in df.columns:
                    df_sorted = df.sort_values("ts")
                else:
                    df_sorted = df.reset_index(drop=True)
                # Clamp new_rows in case of oddities
                take = max(INCREMENT_THRESHOLD, min(new_rows, len(df_sorted)))
                last_batch_df = df_sorted.tail(take)

                # Features/target
                X = df[["year","l_100km"]].replace([np.inf, -np.inf], np.nan).fillna(0)
                y = df["co2_gpkm"].replace([np.inf, -np.inf], np.nan).fillna(0)

                # Train Ridge
                model = Ridge()
                model.fit(X, y)
                preds = model.predict(X)

                r2  = float(r2_score(y, preds))
                mae = float(mean_absolute_error(y, preds))

                # Save EXACT rows used this iteration for the UI
                try:
                    last_batch_df.to_parquet(LAST_BATCH_PARQUET, engine="pyarrow", index=False)
                except Exception as e:
                    print(f"[trainer] Warning: couldn't write last_batch.parquet: {e}")

                # Log metrics (and make it immediate)
                batch_idx += 1
                _append_metrics_csv({
                    "batch": batch_idx,
                    "r2": f"{r2:.4f}",
                    "mae": f"{mae:.4f}",
                    "timestamp": datetime.utcnow().isoformat()
                })

                print(f"[trainer] batch {batch_idx}: rows={len(X)} (+{take}) R2={r2:.3f} MAE={mae:.2f}")

                # Update state
                last_rows = total_rows
                _write_state({"last_rows": last_rows, "batch_idx": batch_idx})

            time.sleep(POLL_SECS)

        except KeyboardInterrupt:
            print("\n[trainer] Stopping…")
            break
        except Exception as e:
            print(f"[trainer] Error: {e}")
            time.sleep(POLL_SECS)

if __name__ == "__main__":
    main()
