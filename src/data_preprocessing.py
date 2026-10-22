import pandas as pd
from pathlib import Path

# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load the sales dataset
DATA_PATH = PROJECT_DIR / "data" / "sales_data.csv"


def load_sales_data():
    df = pd.read_csv(DATA_PATH)

    # Convert date column into datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Sort data by product and date
    df = df.sort_values(["product", "date"])

    # Check for missing values
    print("Dataset loaded successfully!")
    print("Total records:", len(df))
    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    sales_data = load_sales_data()
    print("\nFirst five records:")
    print(sales_data.head())