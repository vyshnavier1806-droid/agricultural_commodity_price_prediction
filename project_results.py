
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Load datasets
frames = []

for filename in ["Agriculture_2023.csv", "Agriculture_2026.csv"]:
    path = DATA / filename
    if path.exists():
        frames.append(pd.read_csv(path))

if not frames:
    raise FileNotFoundError("Dataset CSV files were not found in data/.")

df = pd.concat(frames, ignore_index=True)
df.columns = df.columns.str.strip().str.lower()

df["arrival_date"] = pd.to_datetime(
    df["arrival_date"], errors="coerce"
)

for col in ["min_price", "max_price", "modal_price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["arrival_date", "commodity", "market", "modal_price"]
)
df = df[df["modal_price"] > 0].copy()

# Print summary in terminal
print("\n" + "=" * 55)
print(" AGRICULTURAL COMMODITY PRICE ANALYSIS")
print("=" * 55)
print(f"Total records      : {len(df):,}")
print(f"Unique commodities : {df['commodity'].nunique():,}")
print(f"Unique markets     : {df['market'].nunique():,}")
print(f"Start date         : {df['arrival_date'].min().date()}")
print(f"End date           : {df['arrival_date'].max().date()}")
print(f"Average modal price: Rs. {df['modal_price'].mean():,.2f}")
print(f"Median modal price : Rs. {df['modal_price'].median():,.2f}")

# Prepare dashboard
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle(
    "Agricultural Commodity Price Analysis",
    fontsize=18, fontweight="bold"
)

# 1. Price distribution
ax[0, 0].hist(
    df["modal_price"].clip(upper=df["modal_price"].quantile(0.99)),
    bins=30, edgecolor="white"
)
ax[0, 0].set_title("Modal Price Distribution")
ax[0, 0].set_xlabel("Modal Price (Rs.)")
ax[0, 0].set_ylabel("Number of Records")

# 2. Top commodities by average modal price
top = (
    df.groupby("commodity")["modal_price"]
    .mean()
    .nlargest(10)
    .sort_values()
)
top.plot(kind="barh", ax=ax[0, 1])
ax[0, 1].set_title("Top 10 Commodities by Average Modal Price")
ax[0, 1].set_xlabel("Average Modal Price (Rs.)")
ax[0, 1].set_ylabel("Commodity")

# 3. Minimum, maximum and modal prices
price_sample = df.sample(
    min(500, len(df)), random_state=42
).sort_values("modal_price")

ax[1, 0].scatter(
    price_sample["modal_price"], price_sample["min_price"],
    s=12, alpha=0.5, label="Minimum price"
)
ax[1, 0].scatter(
    price_sample["modal_price"], price_sample["max_price"],
    s=12, alpha=0.5, label="Maximum price"
)
ax[1, 0].set_title("Price Relationship Analysis")
ax[1, 0].set_xlabel("Modal Price (Rs.)")
ax[1, 0].set_ylabel("Price (Rs.)")
ax[1, 0].legend()

# 4. Historical trend for the most frequently recorded commodity
commodity = df["commodity"].value_counts().idxmax()
selected = df[df["commodity"] == commodity]

daily = (
    selected.groupby("arrival_date")["modal_price"]
    .mean()
    .sort_index()
)

ax[1, 1].plot(daily.index, daily.values)
ax[1, 1].set_title(f"Historical Price Trend: {commodity}")
ax[1, 1].set_xlabel("Arrival Date")
ax[1, 1].set_ylabel("Average Modal Price (Rs.)")
ax[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save dashboard for presentation
output_file = OUT / "project_dashboard.png"
plt.savefig(output_file, dpi=200, bbox_inches="tight")

print("\nDashboard saved to:", output_file)
print("Close the graph window after viewing it.")

plt.show()
