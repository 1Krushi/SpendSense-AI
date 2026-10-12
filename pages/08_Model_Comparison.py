
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.style import apply_custom_styles


st.set_page_config(
    page_title="Model Comparison | SpendSense AI",
    page_icon="🧠",
    layout="wide",
)

apply_custom_styles()

st.title("🧠 Sales Forecasting Model Comparison")
st.caption("Compare machine-learning models using historical daily sales data.")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "online_retail.xlsx"


@st.cache_data
def load_daily_sales(path):
    df = pd.read_excel(path)

    required = {"InvoiceNo", "Quantity", "UnitPrice", "InvoiceDate"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing columns: " + ", ".join(sorted(missing))
        )

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    daily = (
        df.set_index("InvoiceDate")["Revenue"]
        .resample("D")
        .sum()
        .asfreq("D", fill_value=0)
        .rename("Revenue")
        .reset_index()
    )

    daily = daily.sort_values("InvoiceDate").reset_index(drop=True)
    return daily


def create_features(data):
    result = data.copy()
    dates = result["InvoiceDate"]

    result["DayOfWeek"] = dates.dt.dayofweek
    result["DayOfMonth"] = dates.dt.day
    result["Month"] = dates.dt.month
    result["TimeIndex"] = np.arange(len(result))

    for lag in [1, 7, 14, 28]:
        result[f"Lag{lag}"] = result["Revenue"].shift(lag)

    result["RollingMean7"] = result["Revenue"].shift(1).rolling(7).mean()
    result["RollingMean28"] = result["Revenue"].shift(1).rolling(28).mean()

    return result.dropna().reset_index(drop=True)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Place online_retail.xlsx inside the project's data folder.")
    st.stop()

try:
    daily = load_daily_sales(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if len(daily) < 60:
    st.error("At least 60 calendar days are needed for this comparison.")
    st.stop()

featured = create_features(daily)

if len(featured) < 20:
    st.error("There are not enough rows after creating forecasting features.")
    st.stop()

feature_columns = [
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "TimeIndex",
    "Lag1",
    "Lag7",
    "Lag14",
    "Lag28",
    "RollingMean7",
    "RollingMean28",
]

# Keep the latest 20% for chronological testing.
split = int(len(featured) * 0.8)
train = featured.iloc[:split]
test = featured.iloc[split:]

X_train = train[feature_columns]
y_train = train["Revenue"]
X_test = test[feature_columns]
y_test = test["Revenue"]

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = np.maximum(model.predict(X_test), 0)
    predictions[name] = predicted

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
        "R2": r2_score(y_test, predicted),
    })

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

st.subheader("Evaluation results")
st.caption(
    "The latest 20% of feature rows are held out for testing. "
    "Scores are based on one-step-ahead predictions using historical lag features."
)

c1, c2, c3 = st.columns(3)

best_rmse = results_df.iloc[0]
best_mae = results_df.loc[results_df["MAE"].idxmin()]
best_r2 = results_df.loc[results_df["R2"].idxmax()]

c1.metric("Lowest RMSE", best_rmse["Model"], f"{best_rmse['RMSE']:,.2f}")
c2.metric("Lowest MAE", best_mae["Model"], f"{best_mae['MAE']:,.2f}")
c3.metric("Highest R²", best_r2["Model"], f"{best_r2['R2']:.3f}")

st.dataframe(
    results_df.style.format({
        "MAE": "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R2": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Compare error metrics")

metric_choice = st.selectbox(
    "Select a metric",
    ["RMSE", "MAE", "R2"],
)

fig = px.bar(
    results_df,
    x="Model",
    y=metric_choice,
    color="Model",
    title=f"Model comparison — {metric_choice}",
    text_auto=".3f",
)

fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Actual vs predicted revenue")

selected_model = st.selectbox(
    "Choose a model",
    list(models.keys()),
)

comparison = pd.DataFrame({
    "Date": test["InvoiceDate"].to_numpy(),
    "Actual Revenue": y_test.to_numpy(),
    "Predicted Revenue": predictions[selected_model],
})

fig2 = px.line(
    comparison,
    x="Date",
    y=["Actual Revenue", "Predicted Revenue"],
    title=f"Actual vs predicted daily revenue — {selected_model}",
)

st.plotly_chart(fig2, use_container_width=True)

st.download_button(
    "Download model comparison CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="spendsense_model_comparison.csv",
    mime="text/csv",
)

st.success(
    "Comparison complete. Choose a model based on its test metrics, "
    "then consider business needs and forecast stability before selecting it."
)
