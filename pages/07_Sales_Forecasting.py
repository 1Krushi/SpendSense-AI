
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.style import apply_custom_styles


st.set_page_config(
    page_title="SpendSense AI | Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

apply_custom_styles()

st.title("📈 Sales Forecasting")
st.caption(
    "Machine-learning estimates based on historical daily revenue"
)

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "online_retail.xlsx"
)


@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_excel(file_path)

    required = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )
    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    df = df.dropna(
        subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    )

    # Remove cancelled invoices and invalid sales.
    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

try:
    df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if df.empty:
    st.warning("No valid sales records were found.")
    st.stop()


# ---------------- DAILY REVENUE ----------------

daily = (
    df.assign(Date=df["InvoiceDate"].dt.normalize())
    .groupby("Date", as_index=False)["Revenue"]
    .sum()
    .sort_values("Date")
)

# Fill missing calendar days with zero sales.
daily = (
    daily.set_index("Date")
    .asfreq("D", fill_value=0)
    .rename_axis("Date")
    .reset_index()
)

if len(daily) < 45:
    st.warning(
        "At least 45 calendar days are recommended for this "
        "forecasting demonstration."
    )
    st.stop()


# ---------------- FEATURES ----------------

def make_features(data):
    result = data.copy()

    result["DayOfWeek"] = result["Date"].dt.dayofweek
    result["DayOfMonth"] = result["Date"].dt.day
    result["Month"] = result["Date"].dt.month
    result["DayOfYear"] = result["Date"].dt.dayofyear
    result["TimeIndex"] = (
        result["Date"] - result["Date"].min()
    ).dt.days

    for lag in [1, 7, 14, 28]:
        result[f"Lag{lag}"] = result["Revenue"].shift(lag)

    result["Rolling7"] = (
        result["Revenue"].shift(1).rolling(7).mean()
    )
    result["Rolling28"] = (
        result["Revenue"].shift(1).rolling(28).mean()
    )

    return result


FEATURES = [
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "DayOfYear",
    "TimeIndex",
    "Lag1",
    "Lag7",
    "Lag14",
    "Lag28",
    "Rolling7",
    "Rolling28",
]

featured = make_features(daily).dropna().reset_index(drop=True)

if len(featured) < 20:
    st.warning("There is insufficient data to train the model.")
    st.stop()


# ---------------- CHOOSE FORECAST HORIZON ----------------

horizon = st.selectbox(
    "Forecast horizon",
    options=[7, 14, 30],
    index=0,
    format_func=lambda x: f"Next {x} days",
)

st.info(
    "This model forecasts overall revenue for the dataset. "
    "It does not forecast individual products or countries."
)


# ---------------- TIME-BASED EVALUATION ----------------

st.subheader("🧪 Model Evaluation")

test_size = max(7, min(30, len(featured) // 5))
train = featured.iloc[:-test_size].copy()
test = featured.iloc[-test_size:].copy()

if len(train) < 10:
    st.warning("There is not enough training history.")
    st.stop()

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(train[FEATURES], train["Revenue"])

test_predictions = np.maximum(
    model.predict(test[FEATURES]), 0
)

mae = mean_absolute_error(
    test["Revenue"], test_predictions
)
rmse = np.sqrt(
    mean_squared_error(test["Revenue"], test_predictions)
)

a, b, c = st.columns(3)

a.metric("Validation MAE", f"£{mae:,.2f}")
b.metric("Validation RMSE", f"£{rmse:,.2f}")
c.metric("Validation Days", f"{len(test):,}")

st.caption(
    "MAE and RMSE measure historical prediction errors in pounds. "
    "Lower values are generally better. Actual future accuracy may differ."
)

evaluation = pd.DataFrame({
    "Date": test["Date"],
    "Actual Revenue": test["Revenue"],
    "Predicted Revenue": test_predictions,
})

evaluation_chart = px.line(
    evaluation,
    x="Date",
    y=["Actual Revenue", "Predicted Revenue"],
    title="Actual vs Predicted Revenue — Held-Out Period",
    template="plotly_white",
)

evaluation_chart.update_layout(height=400)
st.plotly_chart(evaluation_chart, use_container_width=True)


# ---------------- RECURSIVE FUTURE FORECAST ----------------

st.divider()
st.subheader("🔮 Future Revenue Forecast")

history = daily.copy().reset_index(drop=True)
future_rows = []

for _ in range(horizon):
    next_date = history["Date"].iloc[-1] + pd.Timedelta(days=1)

    temporary = pd.concat(
        [
            history,
            pd.DataFrame({
                "Date": [next_date],
                "Revenue": [0.0],
            }),
        ],
        ignore_index=True,
    )

    row_features = make_features(temporary).iloc[[-1]]

    # Historical lag values are available from the prior days.
    predicted = max(
        0.0,
        float(model.predict(row_features[FEATURES])[0])
    )

    future_rows.append({
        "Date": next_date,
        "Forecast Revenue": predicted,
    })

    history = pd.concat(
        [
            history,
            pd.DataFrame({
                "Date": [next_date],
                "Revenue": [predicted],
            }),
        ],
        ignore_index=True,
    )

forecast = pd.DataFrame(future_rows)

historical_chart = daily.rename(
    columns={"Revenue": "Historical Revenue"}
).tail(90)

forecast_chart = px.line(
    forecast,
    x="Date",
    y="Forecast Revenue",
    markers=True,
    title=f"Revenue Forecast — Next {horizon} Days",
    template="plotly_white",
)

forecast_chart.add_scatter(
    x=historical_chart["Date"],
    y=historical_chart["Historical Revenue"],
    mode="lines",
    name="Historical Revenue",
    line=dict(color="#2563eb"),
)

forecast_chart.update_layout(height=430)
st.plotly_chart(forecast_chart, use_container_width=True)


# ---------------- FORECAST SUMMARY ----------------

total_forecast = forecast["Forecast Revenue"].sum()
average_forecast = forecast["Forecast Revenue"].mean()

m1, m2 = st.columns(2)

m1.metric(
    f"Estimated Revenue — {horizon} Days",
    f"£{total_forecast:,.2f}",
)

m2.metric(
    "Estimated Daily Average",
    f"£{average_forecast:,.2f}",
)

st.caption(
    "Forecasts are estimates, not guarantees. The model uses "
    "historical daily revenue and lagged values. Unexpected events, "
    "promotions, seasonality changes, and data limitations can affect accuracy."
)

st.subheader("Forecast Data")

st.dataframe(
    forecast.assign(
        Date=forecast["Date"].dt.strftime("%Y-%m-%d")
    ).round(2),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download Forecast CSV",
    data=forecast.to_csv(index=False).encode("utf-8"),
    file_name="spendsense_sales_forecast.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption("SpendSense AI | Experimental sales forecasting")
