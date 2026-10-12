
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.style import apply_custom_styles


st.set_page_config(
    page_title="SpendSense AI | AI Insights",
    page_icon="🤖",
    layout="wide",
)

apply_custom_styles()

st.title("🤖 AI Customer Insights")
st.caption(
    "Customer behavior analysis using RFM metrics and "
    "unsupervised machine learning"
)

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "online_retail.xlsx"
)


@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_excel(file_path)

    required = [
        "InvoiceNo",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "CustomerID",
        "Description",
        "Country",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
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
    df["CustomerID"] = df["CustomerID"].astype("string")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    df = df.dropna(
        subset=[
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "CustomerID",
        ]
    )

    df = df[
        (df["CustomerID"].str.strip() != "")
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ].copy()

    return df


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

try:
    df = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Could not load data: {error}")
    st.stop()

if df.empty:
    st.warning("No valid customer transactions were found.")
    st.stop()


# ---------------- FILTERS ----------------

st.sidebar.subheader("AI Analysis Filters")

countries = sorted(df["Country"].dropna().unique().tolist())

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=countries,
)

if not selected_countries:
    st.warning("Select at least one country.")
    st.stop()

filtered = df[df["Country"].isin(selected_countries)].copy()

if filtered.empty:
    st.warning("No transactions found for these countries.")
    st.stop()


# ---------------- RFM ANALYSIS ----------------

st.subheader("📊 Customer Behavior Analysis")

# Use one day after the latest transaction as the reference date.
reference_date = filtered["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

rfm = (
    filtered.groupby("CustomerID")
    .agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    .reset_index()
)

rfm["Recency"] = (
    reference_date - rfm["LastPurchase"]
).dt.days

rfm = rfm[
    (rfm["Frequency"] > 0)
    & (rfm["Monetary"] > 0)
].copy()

if rfm.empty:
    st.warning(
        "RFM analysis requires customers with positive sales."
    )
    st.stop()


# ---------------- CUSTOMER CLUSTERING ----------------

if len(rfm) < 3:
    st.warning(
        "At least three customers are needed for customer clustering. "
        "Select more countries or use a larger dataset."
    )
    st.stop()

# Log transformation reduces the effect of highly skewed spending.
features = rfm[["Recency", "Frequency", "Monetary"]].copy()
features["Frequency"] = features["Frequency"].clip(lower=1)
features["Monetary"] = features["Monetary"].clip(lower=0.01)

import numpy as np

model_features = np.log1p(features)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(model_features)

cluster_count = min(4, len(rfm))

model = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10,
)

rfm["Cluster"] = model.fit_predict(scaled_features)

# Assign meaningful names based on each cluster's average RFM profile.
profiles = (
    rfm.groupby("Cluster")
    .agg(
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        Customers=("CustomerID", "count"),
    )
)

profiles["Score"] = (
    profiles["AvgFrequency"].rank(pct=True)
    + profiles["AvgMonetary"].rank(pct=True)
    - profiles["AvgRecency"].rank(pct=True)
)

ranked_clusters = profiles.sort_values("Score", ascending=False).index.tolist()

segment_names = {}

if cluster_count == 1:
    segment_names[ranked_clusters[0]] = "General Customers"
elif cluster_count == 2:
    segment_names[ranked_clusters[0]] = "Higher Engagement"
    segment_names[ranked_clusters[1]] = "Growth Opportunity"
else:
    segment_names[ranked_clusters[0]] = "High-Value Customers"
    segment_names[ranked_clusters[1]] = "Loyal Customers"
    segment_names[ranked_clusters[-1]] = "Needs Attention"

    for cluster in ranked_clusters[2:-1]:
        segment_names[cluster] = "Developing Customers"

rfm["Segment"] = rfm["Cluster"].map(segment_names)

profiles["Segment"] = profiles.index.map(segment_names)


# ---------------- SUMMARY METRICS ----------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Identified Customers", f"{len(rfm):,}")
c2.metric("Customer Segments", f"{cluster_count}")
c3.metric(
    "Average Customer Spend",
    f"£{rfm['Monetary'].mean():,.2f}",
)
c4.metric(
    "Average Recency",
    f"{rfm['Recency'].mean():.1f} days",
)

st.caption(
    "RFM means Recency (days since last purchase), Frequency "
    "(distinct invoices) and Monetary (total recorded sales)."
)

st.divider()


# ---------------- SEGMENT VISUALIZATION ----------------

st.subheader("🧩 Customer Segments")

left, right = st.columns(2)

with left:
    segment_counts = (
        rfm.groupby("Segment", as_index=False)
        .agg(Customers=("CustomerID", "count"))
        .sort_values("Customers", ascending=False)
    )

    fig_segments = px.pie(
        segment_counts,
        names="Segment",
        values="Customers",
        hole=0.55,
        title="Customer Distribution",
        template="plotly_white",
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True,
    )

with right:
    fig_scatter = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color="Segment",
        hover_data=["CustomerID", "Recency"],
        title="Purchase Frequency vs Customer Spend",
        labels={
            "Frequency": "Distinct Invoices",
            "Monetary": "Total Spend (£)",
        },
        template="plotly_white",
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )


# ---------------- SEGMENT PROFILES ----------------

st.subheader("📋 Segment Profiles")

profile_table = (
    rfm.groupby("Segment", as_index=False)
    .agg(
        Customers=("CustomerID", "count"),
        Average_Recency_Days=("Recency", "mean"),
        Average_Invoices=("Frequency", "mean"),
        Average_Spend_GBP=("Monetary", "mean"),
        Total_Revenue_GBP=("Monetary", "sum"),
    )
    .sort_values("Total_Revenue_GBP", ascending=False)
)

st.dataframe(
    profile_table.round(2),
    use_container_width=True,
    hide_index=True,
)


# ---------------- AUTOMATIC RECOMMENDATIONS ----------------

st.divider()
st.subheader("💡 Recommended Actions")

for _, row in profile_table.iterrows():
    segment = row["Segment"]
    count = int(row["Customers"])

    if segment == "High-Value Customers":
        message = (
            "Consider loyalty rewards and personalized product "
            "recommendations to maintain engagement."
        )
    elif segment == "Loyal Customers":
        message = (
            "Test loyalty benefits, complementary products, "
            "and early access to relevant offers."
        )
    elif segment == "Needs Attention":
        message = (
            "Review recent purchase activity and consider a "
            "measured re-engagement campaign."
        )
    elif segment == "Higher Engagement":
        message = (
            "Explore relevant add-on products and customer "
            "appreciation offers."
        )
    else:
        message = (
            "Test relevant promotions and track whether "
            "purchase frequency or spending changes."
        )

    st.info(
        f"**{segment} — {count:,} customers:** {message}"
    )

st.caption(
    "Segment labels are assigned from relative RFM profiles. "
    "K-Means is an unsupervised clustering algorithm; these labels "
    "are analytical categories, not guaranteed predictions of future behavior."
)


# ---------------- CUSTOMER EXPLORER ----------------

st.divider()
st.subheader("🔎 Explore Customer Segments")

chosen_segment = st.selectbox(
    "Select a segment",
    sorted(rfm["Segment"].unique()),
)

segment_customers = (
    rfm[rfm["Segment"] == chosen_segment]
    .sort_values("Monetary", ascending=False)
    .copy()
)

segment_customers["LastPurchase"] = (
    segment_customers["LastPurchase"].dt.strftime("%Y-%m-%d")
)

st.dataframe(
    segment_customers[
        [
            "CustomerID",
            "LastPurchase",
            "Recency",
            "Frequency",
            "Monetary",
            "Segment",
        ]
    ].round(2),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download Customer Analysis",
    data=rfm.to_csv(index=False).encode("utf-8"),
    file_name="spendsense_customer_insights.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption("SpendSense AI | Customer intelligence and RFM analysis")
