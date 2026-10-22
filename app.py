import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sales_data.csv"

st.set_page_config(
    page_title="Supply Chain AI",
    page_icon="📦",
    layout="wide"
)

st.title("📦 AI Supply Chain Demand Forecasting")
st.write("Demand prediction and inventory monitoring dashboard")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Summary metrics
total_sales = int(df["sales"].sum())
total_products = df["product"].nunique()
total_inventory = int(df.groupby("product")["inventory"].last().sum())

col1, col2, col3 = st.columns(3)

col1.metric("Total Units Sold", f"{total_sales:,}")
col2.metric("Products", total_products)
col3.metric("Latest Recorded Inventory", f"{total_inventory:,}")


st.subheader("📅 Filter Sales by Date")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered_df = df[
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
    ]
else:
    filtered_df = df[
        df["date"].dt.date == date_range
    ]

if filtered_df.empty:
    st.warning("No sales records found for this date range.")
else:
    st.metric(
        "Units Sold in Selected Period",
        f"{int(filtered_df['sales'].sum()):,}"
    )
st.subheader("Historical Sales Trends")

product = st.selectbox(
    "Select a product",
    sorted(df["product"].unique())
)

product_data = filtered_df[
    filtered_df["product"] == product
].sort_values("date")

fig = px.line(
    product_data,
    x="date",
    y="sales",
    markers=True,
    title=f"Daily Sales: {product}"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Units Sold"
)


st.plotly_chart(fig, use_container_width=True)

st.subheader("Sales Data")

if product_data.empty:
    st.warning("No sales data for this product in the selected date range.")
else:
    st.dataframe(
        product_data,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="📥 Download Filtered Sales Report",
        data=product_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{product.lower()}_sales_report.csv",
        mime="text/csv"
    )

st.caption(
    "Prototype dashboard using sample data. "
    "Forecasts and inventory recommendations require validation."
)

st.divider()

st.subheader("🤖 AI Demand Prediction")

st.write("Enter product details to estimate future demand.")

with st.form("demand_prediction_form"):
    forecast_product = st.selectbox(
        "Product for prediction",
        sorted(df["product"].unique()),
        key="forecast_product"
    )

    forecast_price = st.number_input(
        "Product price",
        min_value=1.0,
        value=20.0,
        step=1.0
    )

    forecast_promotion = st.selectbox(
        "Promotion active?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    forecast_inventory = st.number_input(
        "Current inventory",
        min_value=0,
        value=180,
        step=10
    )

    predict_button = st.form_submit_button("Predict Demand")


if predict_button:
    from xgboost import XGBRegressor

    import pandas as pd

    MODEL_PATH = PROJECT_DIR / "models" / "demand_model.json"

    if not MODEL_PATH.exists():
        st.error("Trained model not found. Check the models folder.")
    else:
        model = XGBRegressor()
        model.load_model(str(MODEL_PATH))

        # Forecast the day after the latest date in the dataset
        forecast_date = df["date"].max() + pd.Timedelta(days=1)

        # Create input using the same features as model training
        input_data = {
            "price": forecast_price,
            "promotion": forecast_promotion,
            "inventory": forecast_inventory,
            "day": forecast_date.day,
            "weekday": forecast_date.weekday(),
            "product_Biscuits": int(forecast_product == "Biscuits"),
            "product_Milk": int(forecast_product == "Milk"),
            "product_Shampoo": int(forecast_product == "Shampoo"),
            "store_Store_A": 1
        }

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        predicted_units = max(0, round(float(prediction)))

        st.success("Demand prediction completed!")

        col1, col2 = st.columns(2)

        col1.metric("Forecast Demand", f"{predicted_units} units")
        col2.metric("Forecast Date", forecast_date.strftime("%d %b %Y"))

        if predicted_units > forecast_inventory:
            st.warning(
                "Inventory may be insufficient for the predicted demand."
            )
        else:
            st.info(
                "Current inventory is sufficient for the model's "
                "estimated demand."
            )

        st.caption(
            "This is an experimental forecast based on a small sample "
            "dataset. Validate it with more historical data before "
            "making real inventory decisions."
        )
        
st.divider()

st.subheader("⚠️ Inventory Monitoring")

# Get the latest inventory record for each product
latest_data = (
    df.sort_values("date")
    .groupby("product")
    .tail(1)
    .copy()
)

# Calculate average daily sales for each product
average_sales = df.groupby("product")["sales"].mean()

latest_data["Average Daily Sales"] = (
    latest_data["product"].map(average_sales)
)

# Estimate how many days the stock may last
latest_data["Estimated Days of Stock"] = (
    latest_data["inventory"]
    / latest_data["Average Daily Sales"].clip(lower=1)
).round(1)

# Assign stock status
def get_stock_status(days):
    if days < 3:
        return "Critical"
    elif days < 7:
        return "Low"
    else:
        return "Sufficient"

latest_data["Stock Status"] = (
    latest_data["Estimated Days of Stock"]
    .apply(get_stock_status)
)

latest_data["Reorder Recommendation"] = latest_data.apply(
    lambda row: (
        "Reorder soon"
        if row["Estimated Days of Stock"] < 7
        else "Stock looks sufficient"
    ),
    axis=1
)

# Display inventory table

inventory_table = latest_data[
    [
        "product",
        "inventory",
        "Average Daily Sales",
        "Estimated Days of Stock",
        "Stock Status",
        "Reorder Recommendation"
    ]
].rename(
    columns={
        "product": "Product",
        "inventory": "Current Inventory"
    }
)

st.dataframe(
    inventory_table,
    use_container_width=True,
    hide_index=True
)

# Display alerts
critical_count = (
    latest_data["Stock Status"] == "Critical"
).sum()

low_count = (
    latest_data["Stock Status"] == "Low"
).sum()

if critical_count > 0:
    st.error(
        f"🚨 {critical_count} product(s) have critically low stock!"
    )
elif low_count > 0:
    st.warning(
        f"⚠️ {low_count} product(s) have low stock."
    )
else:
    st.success(
        "✅ All products have sufficient estimated stock coverage."
    )
    
st.divider()

st.subheader("📊 Product Sales Comparison")

sales_comparison = (
    df.groupby("product", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
)

fig_comparison = px.bar(
    sales_comparison,
    x="product",
    y="sales",
    color="product",
    text="sales",
    title="Total Units Sold by Product",
    labels={
        "product": "Product",
        "sales": "Total Units Sold"
    }
)

fig_comparison.update_layout(
    xaxis_title="Product",
    yaxis_title="Total Units Sold",
    showlegend=False
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)