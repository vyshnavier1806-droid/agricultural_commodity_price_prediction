
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = OUTPUT_DIR / "combined_cleaned.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["arrival_date"])

# Select the commodity-market series with the most distinct dates
series_counts = (
    df.groupby(["commodity", "market"])["arrival_date"]
    .nunique()
    .sort_values(ascending=False)
)

if series_counts.empty or series_counts.iloc[0] < 20:
    print("Not enough observations for a basic SARIMA experiment.")
    raise SystemExit

commodity, market = series_counts.index[0]

series_df = df[
    (df["commodity"] == commodity) &
    (df["market"] == market)
].copy()

# Prepare daily time series
series_df = (
    series_df.groupby("arrival_date", as_index=False)["modal_price"]
    .mean()
    .sort_values("arrival_date")
)

series = series_df.set_index("arrival_date")["modal_price"]
series = series.asfreq("D")

# Fill missing calendar days by time interpolation for this experiment
series = series.interpolate(method="time").ffill().bfill()

print("SARIMA FORECAST EXPERIMENT")
print("=" * 45)
print("Commodity:", commodity)
print("Market:", market)
print("Available daily dates:", len(series))
print("Date range:", series.index.min(), "to", series.index.max())

# Chronological train-test split
split_index = int(len(series) * 0.8)
train = series.iloc[:split_index]
test = series.iloc[split_index:]

if len(train) < 15 or len(test) < 3:
    print("Insufficient training or test observations.")
    raise SystemExit

# Fit a simple seasonal SARIMA model
try:
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False, maxiter=200)
    predictions = fitted_model.forecast(steps=len(test))
except Exception as error:
    print("SARIMA training failed:", error)
    raise SystemExit

# Evaluate
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print("\nTEST RESULTS")
print("-" * 35)
print(f"Training observations: {len(train)}")
print(f"Testing observations: {len(test)}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save actual vs predicted values
results = pd.DataFrame({
    "date": test.index,
    "actual_modal_price": test.values,
    "predicted_modal_price": predictions.values
})

results.to_csv(
    OUTPUT_DIR / "sarima_predictions.csv",
    index=False
)

# Plot results
plt.figure(figsize=(11, 5))
plt.plot(train.index, train.values, label="Training data")
plt.plot(test.index, test.values, label="Actual test prices")
plt.plot(predictions.index, predictions.values,
         label="SARIMA predictions", linestyle="--")

plt.title(f"SARIMA Forecast: {commodity} - {market}")
plt.xlabel("Date")
plt.ylabel("Modal Price")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sarima_forecast.png")
plt.show()

print("\nSaved predictions to outputs/sarima_predictions.csv")
print("Saved graph to outputs/sarima_forecast.png")
print("\nNote: This is an experimental forecast, not a reliable market forecast.")
