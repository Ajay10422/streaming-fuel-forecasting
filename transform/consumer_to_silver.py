"""
Silver Layer Processor
----------------------
Watches the Bronze directory for new Parquet files, applies cleaning and
transformation logic, and writes the cleaned data to the Silver layer.
This script represents the Bronze-to-Silver ETL step in the pipeline.
"""

import os, time
from glob import glob
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BRONZE_DIR = os.path.join("data", "bronze")
SILVER_DIR = os.path.join("data", "silver")
os.makedirs(SILVER_DIR, exist_ok=True)

# Example mappings (same logic you used in main.py)
brand_mapping = {
    'Toyota': 'Toyota', 'Lexus': 'Toyota', 'Mazda': 'Toyota', 'Subaru': 'Toyota',
    'Volkswagen': 'Volkswagen Group', 'Audi': 'Volkswagen Group', 'Porsche': 'Volkswagen Group',
    'Chevrolet': 'General Motors', 'GMC': 'General Motors', 'Cadillac': 'General Motors',
    'Ford': 'Ford Motor Company', 'Lincoln': 'Ford Motor Company',
    'Honda': 'Honda', 'Acura': 'Honda',
    'Hyundai': 'Hyundai Motor Group', 'Kia': 'Hyundai Motor Group', 'Genesis': 'Hyundai Motor Group',
    'BMW': 'BMW Group', 'MINI': 'BMW Group', 'Rolls-Royce': 'BMW Group',
    'Mercedes-Benz': 'Mercedes-Benz Group', 'Aston Martin': 'Mercedes-Benz Group',
    'Nissan': 'Nissan-Renault Alliance', 'Infiniti': 'Nissan-Renault Alliance',
    'Chrysler': 'Stellantis', 'Dodge': 'Stellantis', 'Jeep': 'Stellantis'
}

fuel_types = {"D", "E", "X", "Z"}

def clean_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules to raw batch (Bronze → Silver)."""
    if df.empty:
        return df

    # Drop rows with missing CO2 or year
    df = df.dropna(subset=["co2_gpkm", "year"])

    # Ensure numeric fields
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["l_100km"] = pd.to_numeric(df["l_100km"], errors="coerce")
    df["co2_gpkm"] = pd.to_numeric(df["co2_gpkm"], errors="coerce")

    # Fill missing values with mean
    df = df.assign(
        l_100km=df["l_100km"].fillna(df["l_100km"].mean()),
        co2_gpkm=df["co2_gpkm"].fillna(df["co2_gpkm"].mean())
    )

    # Map brand
    df["make"] = df["make"].map(brand_mapping).fillna(df["make"])

    # Validate fuel type
    df = df[df["fuel_type"].isin(fuel_types)]

    return df

def main():
    """Monitors the bronze directory and processes new files."""
    processed_files = set()
    print(f"[silver] Watching for new files in {BRONZE_DIR}...")

    while True:
        try:
            # Find all bronze files that haven't been processed yet
            all_bronze_files = set(glob(os.path.join(BRONZE_DIR, "**/*.parquet"), recursive=True))
            new_files = sorted(list(all_bronze_files - processed_files))

            if not new_files:
                time.sleep(3) # Wait before checking again
                continue

            for file_path in new_files:
                print(f"[silver] Processing new bronze file: {file_path}")
                df = pd.read_parquet(file_path)
                df_clean = clean_batch(df)

                if not df_clean.empty:
                    # Use the original file's timestamp for partitioning to maintain consistency
                    now = datetime.utcnow()
                    path = os.path.join(
                        SILVER_DIR,
                        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/hour={now.hour:02d}"
                    )
                    os.makedirs(path, exist_ok=True)
                    out_file = os.path.join(path, os.path.basename(file_path))
                    df_clean.to_parquet(out_file, index=False)
                    print(f"[silver] Wrote {len(df_clean)} rows → {out_file}")

                processed_files.add(file_path)
        except KeyboardInterrupt:
            print("\n[silver] Stopping...")
            break
        except Exception as e:
            print(f"[silver] Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
