
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Load cleaned data
df = pd.read_csv(
    OUTPUT_DIR / "combined_cleaned.csv",
    parse_dates=["arrival_date"]
)

# Create date-based features
df["year"] = df["arrival_date"].dt.year
df["month"] = df["arrival_date"].dt.month
df["day"] = df["arrival_date"].dt.day

# Input features
categorical_features = [
    "state", "district", "market",
    "commodity", "variety", "grade"
]

numeric_features = [
    "year", "month", "day",
    "min_price", "max_price"
]

features = categorical_features + numeric_features
target = "modal_price"

# Remove rows with missing feature values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess categorical data
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Build Decision Tree model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=5,
                random_state=42
            )
        )
    ]
)

# Train model
print("Training Decision Tree Regression model...")
model.fit(X_train, y_train)

# Predict prices
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-" * 35)
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# Save trained model
model_path = OUTPUT_DIR / "decision_tree_model.joblib"
joblib.dump(model, model_path)

# Save sample predictions
results = pd.DataFrame({
    "Actual_Modal_Price": y_test.to_numpy(),
    "Predicted_Modal_Price": predictions
})

results.to_csv(
    OUTPUT_DIR / "decision_tree_predictions.csv",
    index=False
)

print("\nModel saved to:", model_path)
print("Predictions saved to outputs/decision_tree_predictions.csv")
