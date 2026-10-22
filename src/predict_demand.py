import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "demand_model.json"


def predict_demand():
    # Load the trained model
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    # Example future demand inputs
    future_data = pd.DataFrame([
        {
            "price": 20,
            "promotion": 0,
            "inventory": 180,
            "day": 8,
            "weekday": 0,
            "product_Biscuits": 1,
            "product_Milk": 0,
            "product_Shampoo": 0,
            "store_Store_A": 1
        }
    ])

    # Generate prediction
    prediction = model.predict(future_data)[0]

    print("Product: Biscuits")
    print("Forecast demand:", max(0, round(float(prediction))), "units")


if __name__ == "__main__":
    predict_demand()