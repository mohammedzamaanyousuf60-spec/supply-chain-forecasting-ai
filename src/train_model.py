
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "sales_data.csv"
MODEL_DIR = PROJECT_DIR / "models"

def train_model():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product", "date"])

    # Create calendar features
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.dayofweek

    # Convert product and store names into numeric features
    df = pd.get_dummies(df, columns=["product", "store"], dtype=int)

    # Use the earlier dates for training and later dates for testing
    df = df.sort_values("date").reset_index(drop=True)

    features = [
        col for col in df.columns
        if col not in ["date", "sales"]
    ]

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[features]
    y_train = train["sales"]
    X_test = test[features]
    y_test = test["sales"]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(str(MODEL_DIR / "demand_model.json"))

    print("Model training completed!")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print("Model saved to models/demand_model.json")

if __name__ == "__main__":
    train_model()