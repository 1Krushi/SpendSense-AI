
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Customer Segmentation | SpendSense AI",
    page_icon="👥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "online_retail.xlsx"

st.title("👥 Customer Segmentation AI")
st.caption("Discover customer groups using Recency, Frequency and Monetary (RFM) analysis.")

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.18);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    required = [
        "InvoiceNo", "InvoiceDate", "CustomerID",
        "Quantity", "UnitPrice", "Description"
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = df["CustomerID"].astype("string").str.replace(r"\.0$", "", regex=True)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df = df.dropna(subset=["InvoiceDate", "CustomerID", "Quantity", "UnitPrice"])
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


if not DATA_FILE.exists():
    st.error("Dataset not found. Place online_retail.xlsx inside the data folder.")
    st.stop()

try:
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if df.empty:
    st.warning("No valid transactions were found in the dataset.")
    st.stop()

st.sidebar.header("Segmentation Settings")
n_clusters = st.sidebar.slider(
    "Number of customer groups",
    min_value=2,
    max_value=6,
    value=4
)

st.sidebar.caption(
    "RFM means Recency (how recently a customer purchased), "
    "Frequency (number of purchases) and Monetary (total spend)."
)

# Build customer-level RFM metrics
analysis_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

rfm["Recency"] = (analysis_date - rfm["LastPurchase"]).dt.days
rfm = rfm[
    (rfm["Frequency"] > 0)
    & (rfm["Monetary"] > 0)
].copy()

if len(rfm) < n_clusters:
    st.warning(
        f"There are only {len(rfm)} customers available. "
        "Choose fewer customer groups in the sidebar."
    )
    st.stop()

# Log-transform spending and frequency to reduce skew
features = rfm[["Recency", "Frequency", "Monetary"]].copy()
features["Frequency"] = features["Frequency"].clip(lower=1)
features["Monetary"] = features["Monetary"].clip(lower=0.01)
features = features.apply(lambda column: __import__("numpy").log1p(column))

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
rfm["Cluster"] = model.fit_predict(scaled_features)

# Give clusters readable names using relative spending and purchase activity
cluster_summary = rfm.groupby("Cluster").agg(
    Customers=("CustomerID", "nunique"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    TotalRevenue=("Monetary", "sum")
).reset_index()

monetary_median = cluster_summary["AvgMonetary"].median()
frequency_median = cluster_summary["AvgFrequency"].median()

def name_segment(row):
    if row["AvgMonetary"] >= monetary_median and row["AvgFrequency"] >= frequency_median:
        return "VIP Customers"
    if row["AvgFrequency"] >= frequency_median:
        return "Loyal Customers"
    if row["AvgRecency"] <= cluster_summary["AvgRecency"].median():
        return "Recent Customers"
    return "Needs Attention"

cluster_summary["Segment"] = cluster_summary.apply(name_segment, axis=1)
segment_names = dict(zip(cluster_summary["Cluster"], cluster_summary["Segment"]))
rfm["Segment"] = rfm["Cluster"].map(segment_names)

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers Analysed", f"{len(rfm):,}")
c2.metric("Customer Groups", f"{n_clusters}")
c3.metric("Average Customer Spend", f"£{rfm['Monetary'].mean():,.2f}")
c4.metric("Highest Customer Spend", f"£{rfm['Monetary'].max():,.2f}")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Segment Overview",
    "🔎 Customer Explorer",
    "💡 Business Recommendations"
])

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Customer Distribution")
        segment_counts = (
            rfm["Segment"].value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            segment_counts,
            names="Segment",
            values="Customers",
            hole=0.55,
            template="plotly_white",
            title="Customers by Segment"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Customer Value")
        fig = px.bar(
            cluster_summary.sort_values("TotalRevenue", ascending=False),
            x="Segment",
            y="TotalRevenue",
            color="Segment",
            template="plotly_white",
            title="Total Revenue by Customer Group",
            labels={"TotalRevenue": "Revenue (£)", "Segment": "Customer Group"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Performance")
    display_summary = cluster_summary.rename(columns={
        "Customers": "Number of Customers",
        "AvgRecency": "Average Recency (days)",
        "AvgFrequency": "Average Orders",
        "AvgMonetary": "Average Spend (£)",
        "TotalRevenue": "Total Spend (£)"
    })
    st.dataframe(
        display_summary.drop(columns=["Cluster"]).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Segment names are heuristic labels based on this dataset's cluster averages. "
        "They are not guaranteed predictions of future customer behavior."
    )

with tab2:
    st.subheader("Explore Individual Customers")

    segment_options = ["All"] + sorted(rfm["Segment"].unique().tolist())
    chosen_segment = st.selectbox("Filter by customer group", segment_options)

    customer_view = rfm.copy()
    if chosen_segment != "All":
        customer_view = customer_view[customer_view["Segment"] == chosen_segment]

    search_customer = st.text_input("Search Customer ID")
    if search_customer.strip():
        customer_view = customer_view[
            customer_view["CustomerID"].str.contains(
                search_customer.strip(), case=False, na=False
            )
        ]

    st.dataframe(
        customer_view[[
            "CustomerID", "Segment", "Recency",
            "Frequency", "Monetary"
        ]].sort_values("Monetary", ascending=False).rename(columns={
            "CustomerID": "Customer ID",
            "Recency": "Days Since Last Purchase",
            "Frequency": "Number of Orders",
            "Monetary": "Total Spend (£)"
        }).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Export Customer Segments",
        data=customer_view.to_csv(index=False).encode("utf-8"),
        file_name="spendsense_customer_segments.csv",
        mime="text/csv"
    )

    st.subheader("Recency vs. Monetary Value")
    fig = px.scatter(
        customer_view,
        x="Recency",
        y="Monetary",
        size="Frequency",
        color="Segment",
        hover_name="CustomerID",
        template="plotly_white",
        labels={
            "Recency": "Days Since Last Purchase",
            "Monetary": "Total Spend (£)",
            "Frequency": "Number of Orders"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Recommended Actions")

    recommendations = {
        "VIP Customers": (
            "Reward valuable repeat customers with loyalty benefits, early access "
            "to new products and personalised recommendations."
        ),
        "Loyal Customers": (
            "Encourage repeat purchases with loyalty points, bundles and relevant "
            "cross-selling offers."
        ),
        "Recent Customers": (
            "Welcome recent buyers with useful product suggestions and a smooth "
            "follow-up shopping experience."
        ),
        "Needs Attention": (
            "Test a friendly re-engagement campaign, product recommendations or "
            "a limited promotion to encourage another purchase."
        )
    }

    for _, row in cluster_summary.sort_values("TotalRevenue", ascending=False).iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['Segment']}")
            st.write(recommendations.get(row["Segment"], "Review this group's purchasing patterns."))
            st.caption(
                f"{int(row['Customers']):,} customers · "
                f"Average spend £{row['AvgMonetary']:,.2f} · "
                f"Average recency {row['AvgRecency']:.0f} days"
            )

st.divider()
st.caption("SpendSense AI | Customer analytics powered by RFM analysis and K-Means clustering")
