"""
Producer
--------
Reads rows from Cars_Cleaned.xlsx and streams them into Kafka as JSON.

- Acts as the "entry point" (Bronze layer).
- Converts each row into an event dict (with timestamp, make, model, year, etc.).
- Sends each event to Kafka topic (default: "energuide-events").
- Messages are stored durably in Kafka and can be replayed later.
"""
"""
Producer that streams 100 rows every 5 seconds from Cars_Cleaned.xlsx into Kafka.
"""

import os, json, time
from datetime import datetime, timezone
import pandas as pd
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

TOPIC = os.getenv("KAFKA_TOPIC", "energuide-events")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

def build_producer(bootstrap: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        key_serializer=lambda k: k.encode() if isinstance(k, str) else k,
        value_serializer=lambda v: json.dumps(v).encode(),
        acks="all"
    )

def row_to_event(row: pd.Series) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "make": str(row.get("Make", "")),
        "model": str(row.get("Model", "")),
        "year": int(row.get("Model year", 0)),
        "transmission_code": str(row.get("Transmission", "")),
        "fuel_type": str(row.get("Fuel type", "")),
        "co2_gpkm": float(row.get("CO2 emissions (g/km)", 0.0) or 0.0),
        "l_100km": float(row.get("Combined (L/100 km)", 0.0) or 0.0),
        "province": "ON"
    }

def main():
    df = pd.read_excel(os.path.join("data", "Cars_Cleaned.xlsx"))
    producer = build_producer(BOOTSTRAP)

    print(f"[producer] Starting to send data indefinitely...")
    batch_size = 2 # Number of rows to send in each batch

    try:
        while True:
            # Sample random rows from the dataframe to simulate a continuous stream
            batch = df.sample(n=batch_size, replace=True)
            for _, row in batch.iterrows():
                evt = row_to_event(row)
                key = f"{evt['make']}-{evt['model']}-{evt['year']}"
                producer.send(TOPIC, key=key, value=evt)
            producer.flush()
            print(f"[producer] Sent {len(batch)} new rows → Kafka")
            time.sleep(2)  # Wait before sending the next batch
    except KeyboardInterrupt:
        print("\n[producer] Stopping...")

if __name__ == "__main__":
    main()
