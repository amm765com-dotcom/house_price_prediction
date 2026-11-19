"""
Generate synthetic Lahore house price rows into SQLite for testing/training.

Usage:
    python 02_ADD_SAMPLE_DATA.py             # adds 500 rows if table has < 50 rows
    python 02_ADD_SAMPLE_DATA.py 1000        # generates 1000 rows
    python 02_ADD_SAMPLE_DATA.py 500 --force # adds rows even if >= 50 rows exist
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 500
FORCE = "--force" in sys.argv
DB_PATH = "house_prices.db"
SKIP_IF_ROWS = 50  # skip if table already has enough rows to avoid accidental duplicates

# Matches location rates and property multipliers used in app.py and index.html
LOC_RATE = {
    "Dha": 20000,
    "Gulberg": 18000,
    "Model Town": 16000,
    "Bahria Town": 14000,
    "Johar Town": 13000,
    "Other": 10000,
}
TYPE_MULT = {"House": 1.0, "Apartment": 0.85, "Villa": 1.3}


def make_rows(n, seed=7):
    rng = np.random.default_rng(seed)
    loc = rng.choice(list(LOC_RATE), n)
    typ = rng.choice(list(TYPE_MULT), n, p=[0.6, 0.25, 0.15])
    area = np.round(rng.uniform(3, 40, n) * 272.25, 1)  # 3 to 40 marla in sq ft
    bed = np.clip((area / 450).astype(int) + rng.integers(0, 2, n), 1, 8)
    bath = np.clip(bed - rng.integers(0, 2, n), 1, 7).astype(float)
    floors = np.where(typ == "Apartment", 1, rng.integers(1, 4, n))
    year = rng.integers(1985, 2027, n)

    rate = np.array([LOC_RATE[l] for l in loc])
    mult = np.array([TYPE_MULT[t] for t in typ])
    age_f = np.maximum(0.8, 1 - 0.006 * (2026 - year))

    price = (
        area * rate * mult * age_f * (1 + 0.04 * (floors - 1))
        + bed * 250000
        + bath * 150000
    )
    price = np.round(price * rng.normal(1.0, 0.06, n), -3)

    return pd.DataFrame({
        "area": area,
        "bedrooms": bed,
        "bathrooms": bath,
        "location": loc,
        "property_type": typ,
        "number_floors": floors,
        "year_built": year,
        "price": price,
    })


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        try:
            current = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
        except sqlite3.OperationalError:
            current = 0

        if current >= SKIP_IF_ROWS and not FORCE:
            print(f"Table already has {current} rows. Nothing added (use --force to add anyway).")
            return

        df = make_rows(N_ROWS)
        valid_mask = (df.area > 0) & (df.price > 0) & (df.bedrooms >= 0) & (df.bathrooms >= 0)
        df = df[valid_mask].drop_duplicates()

        df.to_sql("properties", conn, if_exists="append", index=False)
        total = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
    finally:
        conn.close()

    print(f"Added {len(df)} generated rows. Table now has {total} rows.")


if __name__ == "__main__":
    main()