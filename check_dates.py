
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "outputs" / "combined_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

df = pd.read_csv(DATA_FILE, parse_dates=["arrival_date"])

print("=" * 50)
print("AGRICULTURAL COMMODITY DATA COVERAGE")
print("=" * 50)

print("Total records:", len(df))
print("First date:", df["arrival_date"].min())
print("Last date:", df["arrival_date"].max())
print("Commodities:", df["commodity"].nunique())
print("Markets:", df["market"].nunique())

coverage = (
    df.dropna(subset=["arrival_date"])
    .groupby(["commodity", "market"])
    .agg(
        records=("arrival_date", "size"),
        unique_dates=("arrival_date", "nunique"),
        first_date=("arrival_date", "min"),
        last_date=("arrival_date", "max")
    )
    .reset_index()
    .sort_values("unique_dates", ascending=False)
)

coverage.to_csv(
    OUTPUT_DIR / "commodity_market_coverage.csv",
    index=False
)

print("\nTop 10 commodity-market combinations:")
print(coverage.head(10).to_string(index=False))

print(
    "\nCombinations with at least 30 actual observed dates:",
    (coverage["unique_dates"] >= 30).sum()
)

print("\nReport saved to outputs/commodity_market_coverage.csv")
