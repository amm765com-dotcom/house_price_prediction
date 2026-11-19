"""
ETL + EDA pipeline for Lahore house prices.

How to run:
    python 01_EXTRACT.py                    # uses built-in sample rows
    python 01_EXTRACT.py lahore_data.csv    # uses your own CSV file

Required CSV columns: area (sq ft), bedrooms, bathrooms, location, property_type, price
Optional CSV columns: number_floors, year_built
Outputs: house_prices.db (table: properties) and eda_report.png
"""

import sqlite3
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # saves image without popping up a GUI window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field, ValidationError


DB_FILE = "house_prices.db"
EXPECTED_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "location",
    "property_type",
    "number_floors",
    "year_built",
    "price",
]


class PropertyRecord(BaseModel):
    """Schema to check each row before saving to the database."""
    area: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    location: str = Field(..., min_length=1)
    property_type: str = Field(..., min_length=1)
    number_floors: Optional[int] = Field(default=1, ge=1)
    year_built: Optional[int] = Field(default=2000, ge=1800)
    price: float = Field(..., gt=0)


def extract_rows(csv_file=None):
    """Load data from a CSV file, or fall back to sample rows for testing."""
    if csv_file:
        print(f"--- [EXTRACT] Reading {csv_file} ---")
        return pd.read_csv(csv_file)

    print("--- [EXTRACT] No CSV passed, using sample demo rows ---")
    sample_defaults = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "number_floors": 2,
        "year_built": 2018,
    }

    sample_rows = [
        {**sample_defaults, "area": 1200.5, "location": "Gulberg", "property_type": "House", "price": 15500000.0},
        {"area": 850.0, "bedrooms": 2, "bathrooms": 1.0, "location": "DHA", "property_type": "Apartment", "number_floors": 1, "year_built": 2020, "price": 9000000.0},
        {"area": 1500.0, "bedrooms": 4, "bathrooms": 3.0, "location": "Johar Town", "property_type": "House", "number_floors": 2, "year_built": 2016, "price": 17500000.0},
        {**sample_defaults, "area": 1200.5, "location": "Gulberg", "property_type": "House", "price": 15500000.0},  # duplicate row
        {"area": -500.0, "bedrooms": 4, "bathrooms": 3.0, "location": "Model Town", "property_type": "House", "number_floors": 2, "year_built": 2015, "price": -20000.0},  # negative values
        {"area": None, "bedrooms": 4, "bathrooms": 3.0, "location": "Bahria Town", "property_type": "Villa", "number_floors": 2, "year_built": 2021, "price": 22000000.0},  # missing area
    ]
    return pd.DataFrame(sample_rows)


def clean_data(df):
    """Clean raw rows: drop missing/duplicates, normalize text, and fill defaults."""
    print("--- [TRANSFORM] Cleaning data ---")
    start_count = len(df)
    df = df.copy()

    # Make sure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[EXPECTED_COLUMNS]

    # Drop rows missing core fields
    required_cols = ["area", "price", "bedrooms", "bathrooms", "location", "property_type"]
    df = df.dropna(subset=required_cols)

    # Remove duplicate rows
    df = df.drop_duplicates().copy()

    # Clean text fields
    df["location"] = df["location"].astype(str).str.strip().str.title()
    df["property_type"] = df["property_type"].astype(str).str.strip().str.title()

    # Fill optional fields with sensible defaults
    df["number_floors"] = df["number_floors"].fillna(1)
    df["year_built"] = df["year_built"].fillna(2000)

    # Filter out extreme price outliers using IQR only when dataset is large enough
    if len(df) >= 30:
        q1, q3 = df["price"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        df = df[df["price"].between(lower_bound, upper_bound)]

    end_count = len(df)
    print(f"Rows: {start_count} -> {end_count} (removed {start_count - end_count})")
    return df


def validate_rows(df):
    """Validate each row with Pydantic and keep only valid records."""
    print("--- [VALIDATE] Checking rows with Pydantic ---")
    valid_rows = []
    rejected_count = 0

    for idx, row in df.iterrows():
        try:
            record = PropertyRecord(**row.to_dict())
            valid_rows.append(record.model_dump())
        except ValidationError as err:
            rejected_count += 1
            bad_fields = [item["loc"][0] for item in err.errors()]
            print(f"  Skipped row {idx} (issue in: {bad_fields})")

    print(f"Valid: {len(valid_rows)} | Rejected: {rejected_count}")
    return pd.DataFrame(valid_rows)


def save_to_db(df):
    """Write cleaned and validated rows into SQLite."""
    print("--- [LOAD] Saving to SQLite ---")
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("properties", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows into '{DB_FILE}' (table: properties)")


def run_eda():
    """Print basic summary stats and save 3 quick EDA charts."""
    print("\n--- [EDA] Exploratory Data Analysis ---")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM properties", conn)
    conn.close()

    if df.empty:
        print("Database table is empty, skipping EDA.")
        return

    print(df.describe().round(1))

    print("\nCorrelation with price:")
    numeric_df = df.select_dtypes("number")
    print(numeric_df.corr()["price"].sort_values(ascending=False).round(3))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Price distribution
    sns.histplot(df["price"] / 1e6, kde=len(df) > 2, color="#0f6b55", ax=axes[0])
    axes[0].set(title="Price distribution", xlabel="Price (million PKR)")

    # 2. Area vs Price
    sns.scatterplot(x="area", y=df["price"] / 1e6, data=df, hue="location", ax=axes[1])
    axes[1].set(title="Area vs price", xlabel="Area (sq ft)", ylabel="Price (million PKR)")

    # 3. Average price by location
    avg_by_loc = (df.groupby("location")["price"].mean() / 1e6).sort_values()
    avg_by_loc.plot.barh(color="#e0a22d", ax=axes[2])
    axes[2].set(title="Average price by location", xlabel="Million PKR")

    plt.tight_layout()
    plt.savefig("eda_report.png", dpi=130)
    print("\nSaved charts to eda_report.png")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None

    raw_df = extract_rows(csv_arg)
    cleaned_df = clean_data(raw_df)
    validated_df = validate_rows(cleaned_df)

    if validated_df.empty:
        print("No valid rows left after validation.")
    else:
        save_to_db(validated_df)
        run_eda()

    print("\nDone.")