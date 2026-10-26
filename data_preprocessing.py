
import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load both datasets
df_2023 = pd.read_csv(DATA_DIR / "Agriculture_2023.csv")
df_2026 = pd.read_csv(DATA_DIR / "Agriculture_2026.csv")

print("2023 dataset shape:", df_2023.shape)
print("2026 dataset shape:", df_2026.shape)

# Standardize column names
def standardize_columns(df):
    df = df.copy()
    df.columns = [
        str(col).strip().lower()
        .replace("_x0020_", "_")
        .replace(" ", "_")
        for col in df.columns
    ]
    return df

df_2023 = standardize_columns(df_2023)
df_2026 = standardize_columns(df_2026)

# Combine the datasets
df = pd.concat([df_2023, df_2026], ignore_index=True)

# Display column names to verify the dataset structure
print("\nColumn names:")
print(df.columns.tolist())

# Convert date column
df["arrival_date"] = pd.to_datetime(
    df["arrival_date"], format="%Y-%m-%d", errors="coerce"
)

# Convert prices to numeric values
price_columns = ["min_price", "max_price", "modal_price"]

for col in price_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing essential values
df = df.dropna(
    subset=["arrival_date", "commodity", "min_price",
            "max_price", "modal_price"]
)

# Remove duplicate rows
df = df.drop_duplicates()

# Validate price ranges
valid_prices = (
    (df["min_price"] >= 0)
    & (df["max_price"] >= df["min_price"])
    & (df["modal_price"] >= df["min_price"])
    & (df["modal_price"] <= df["max_price"])
)

invalid_count = int((~valid_prices).sum())
print("\nRows with invalid prices:", invalid_count)

df = df[valid_prices].copy()

# Save the cleaned dataset
output_path = OUTPUT_DIR / "combined_cleaned.csv"
df.to_csv(output_path, index=False)

# Display summary
print("\nData preprocessing completed!")
print("Cleaned dataset shape:", df.shape)
print("Date range:", df["arrival_date"].min(),
      "to", df["arrival_date"].max())
print("Unique commodities:", df["commodity"].nunique())

if "market" in df.columns:
    print("Unique markets:", df["market"].nunique())

print("Saved cleaned dataset to:", output_path)
