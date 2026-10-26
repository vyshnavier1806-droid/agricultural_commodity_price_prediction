
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
file_path = OUTPUT_DIR / "combined_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=["arrival_date"])

print("AGRICULTURAL COMMODITY PRICE ANALYSIS")
print("=" * 45)

# Basic information
print("\nDataset shape:", df.shape)
print("Number of commodities:", df["commodity"].nunique())
print("Number of markets:", df["market"].nunique())
print("Date range:", df["arrival_date"].min(),
      "to", df["arrival_date"].max())

# Descriptive statistics
print("\nPRICE DESCRIPTIVE STATISTICS")
print(df[["min_price", "max_price", "modal_price"]].describe())

# Average modal price by commodity
commodity_prices = (
    df.groupby("commodity")["modal_price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTOP 10 COMMODITIES BY AVERAGE MODAL PRICE")
print(commodity_prices.head(10).round(2))

# Save statistics
commodity_prices.to_csv(
    OUTPUT_DIR / "commodity_average_prices.csv",
    header=["average_modal_price"]
)

# Graph 1: Distribution of modal prices
plt.figure(figsize=(10, 5))
sns.histplot(df["modal_price"], bins=40, kde=True)
plt.title("Distribution of Agricultural Modal Prices")
plt.xlabel("Modal Price")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "modal_price_distribution.png")
plt.show()

# Graph 2: Top 10 commodities by average modal price
plt.figure(figsize=(11, 6))
commodity_prices.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Commodities by Average Modal Price")
plt.xlabel("Average Modal Price")
plt.ylabel("Commodity")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_10_commodities.png")
plt.show()

# Graph 3: Correlation between prices
plt.figure(figsize=(7, 5))
correlation = df[
    ["min_price", "max_price", "modal_price"]
].corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Agricultural Prices")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_correlation.png")
plt.show()

print("\nEDA completed successfully!")
print("Statistics and graphs saved in the outputs folder.")
