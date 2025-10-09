"""
Consumer to Parquet (Bronze Writer)
-----------------------------------
Consumes Kafka messages and writes them into partitioned Parquet files.

- This is the start of the ETL process:
  Producer -> Kafka -> Bronze storage (Parquet).
- Parquet is a **columnar file format**:
    - Stores data column by column instead of row by row (like CSV).
    - Much faster for analytics queries (e.g., only scan 'co2_gpkm').
    - Smaller file size due to compression.
    - Schema-aware: keeps data types clean (int, float, string).
- Files are partitioned by time (year, month, day, hour) for efficient querying later.
"""
import os, json, signal, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from kafka import KafkaConsumer
from dotenv import load_dotenv

# Minimal schema guard (keeps bronze consistent)
REQUIRED = ["ts","make","model","year","transmission_code","fuel_type","co2_gpkm","l_100km","province"]

def valid(evt: dict) -> bool:
    if not isinstance(evt, dict):
        return False
    return all(k in evt for k in REQUIRED)

def main():
    load_dotenv()
    topic = os.getenv("KAFKA_TOPIC", "energuide-events")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

    # Where bronze files land
    BRONZE_ROOT = Path("data/bronze")
    BRONZE_ROOT.mkdir(parents=True, exist_ok=True)

    # Kafka consumer (its own group so it doesn’t share offsets with your print consumer)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="bronze-writer",
        key_deserializer=lambda k: k.decode() if k else None,
        value_deserializer=lambda v: json.loads(v.decode()),
    )

    print(f"[bronze] listening on {topic} @ {bootstrap}")
    buffer = []
    BATCH_SIZE = 10  # write every N messages (tweak as you like)

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return

        # Use the timestamp of the *first event* in the batch for partitioning
        # This ensures data is stored based on event time, not processing time.
        try:
            first_event_ts_str = buffer[0].get("ts")
            partition_time = datetime.fromisoformat(first_event_ts_str)
        except (IndexError, TypeError, ValueError):
            partition_time = datetime.now(timezone.utc) # Fallback

        out_dir = BRONZE_ROOT / f"year={partition_time:%Y}" / f"month={partition_time:%m}" / f"day={partition_time:%d}" / f"hour={partition_time:%H}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"batch_{partition_time:%Y%m%dT%H%M%S}.parquet"

        pd.DataFrame(buffer).to_parquet(out_path, engine="pyarrow", index=False)
        print(f"[bronze] wrote {len(buffer)} rows → {out_path}")
        buffer = []

    # Graceful shutdown (Ctrl+C flushes last batch)
    def _graceful(*_):
        print("\n[bronze] stopping… flushing remaining records")
        flush_buffer()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _graceful)

    # Stream loop
    for msg in consumer:
        evt = msg.value
        if valid(evt):
            buffer.append(evt)
        # else: drop silently; bronze is allowed to be raw, but we still keep required keys

        if len(buffer) >= BATCH_SIZE:
            flush_buffer()

if __name__ == "__main__":
    main()
