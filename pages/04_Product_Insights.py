
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Product Insights | SpendSense AI",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "online_retail.xlsx"

st.title("📦 Product & Sales Intelligence")
st.caption("Explore product performance, sales trends and geographic revenue.")

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.18);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    required = [
        "InvoiceNo", "InvoiceDate", "CustomerID",
        "Description", "Quantity", "UnitPrice", "Country"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df = df.dropna(subset=[
        "InvoiceNo", "InvoiceDate", "Description",
        "Quantity", "UnitPrice", "Country"
    ]).copy()

    # Exclude cancellations, returned quantities and invalid prices
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Year"] = df["InvoiceDate"].dt.year
    return df


if not DATA_FILE.exists():
    st.error("Dataset not found. Put online_retail.xlsx inside the data folder.")
    st.stop()

try:
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if df.empty:
    st.warning("No valid sales records were found.")
    st.stop()

# Sidebar filters
st.sidebar.header("Sales Filters")

countries = sorted(df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=countries
)

years = sorted(df["Year"].unique().tolist())
selected_years = st.sidebar.multiselect(
    "Years",
    options=years,
    default=years
)

top_n = st.sidebar.slider(
    "Number of products to display",
    min_value=5,
    max_value=25,
    value=10
)

if selected_countries and selected_years:
    filtered = df[
        df["Country"].isin(selected_countries)
        & df["Year"].isin(selected_years)
    ].copy()
else:
    filtered = df.iloc[0:0].copy()

if filtered.empty:
    st.warning("No records match your filters. Select at least one country and year.")
    st.stop()

# KPI calculations
total_revenue = filtered["Revenue"].sum()
total_units = filtered["Quantity"].sum()
total_orders = filtered["InvoiceNo"].nunique()
unique_products = filtered["Description"].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"£{total_revenue:,.2f}")
k2.metric("Units Sold", f"{total_units:,.0f}")
k3.metric("Orders", f"{total_orders:,}")
k4.metric("Products Sold", f"{unique_products:,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Product Performance",
    "🌍 Country Analysis",
    "📈 Sales Trends",
    "📋 Detailed Data"
])

with tab1:
    st.subheader("Top Products by Revenue")

    product_revenue = (
        filtered.groupby("Description", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            UnitsSold=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique")
        )
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        product_revenue.sort_values("Revenue"),
        x="Revenue",
        y="Description",
        orientation="h",
        template="plotly_white",
        color="Revenue",
        color_continuous_scale="Blues",
        title=f"Top {top_n} Products by Revenue",
        labels={"Revenue": "Revenue (£)", "Description": "Product"}
    )
    fig.update_layout(yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Best-selling products by quantity")
        top_quantity = (
            filtered.groupby("Description", as_index=False)["Quantity"]
            .sum()
            .sort_values("Quantity", ascending=False)
            .head(top_n)
        )

        fig = px.bar(
            top_quantity.sort_values("Quantity"),
            x="Quantity",
            y="Description",
            orientation="h",
            template="plotly_white",
            title="Top Products by Units Sold",
            labels={"Quantity": "Units Sold", "Description": "Product"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Product revenue share")
        fig = px.pie(
            product_revenue,
            names="Description",
            values="Revenue",
            hole=0.45,
            template="plotly_white",
            title=f"Revenue Share of Top {top_n} Products"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Product performance table")
    st.dataframe(
        product_revenue.rename(columns={
            "Description": "Product",
            "Revenue": "Revenue (£)",
            "UnitsSold": "Units Sold",
            "Orders": "Number of Orders"
        }).round(2),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("Geographic Sales Performance")

    country_sales = (
        filtered.groupby("Country", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum")
        )
        .sort_values("Revenue", ascending=False)
    )

    fig = px.bar(
        country_sales.head(15).sort_values("Revenue"),
        x="Revenue",
        y="Country",
        orientation="h",
        color="Revenue",
        color_continuous_scale="Teal",
        template="plotly_white",
        title="Top 15 Countries by Revenue",
        labels={"Revenue": "Revenue (£)", "Country": "Country"}
    )
    fig.update_layout(yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Country performance table")
    st.dataframe(
        country_sales.rename(columns={
            "Country": "Country",
            "Revenue": "Revenue (£)",
            "Orders": "Orders",
            "UnitsSold": "Units Sold"
        }).round(2),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("Revenue Over Time")

    monthly_sales = (
        filtered.groupby("Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum")
        )
        .sort_values("Month")
    )

    fig = px.line(
        monthly_sales,
        x="Month",
        y="Revenue",
        markers=True,
        template="plotly_white",
        title="Monthly Revenue Trend",
        labels={"Revenue": "Revenue (£)", "Month": "Month"}
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            monthly_sales,
            x="Month",
            y="Orders",
            template="plotly_white",
            title="Monthly Order Volume"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            monthly_sales,
            x="Month",
            y="UnitsSold",
            template="plotly_white",
            title="Monthly Units Sold"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Monthly sales summary")
    st.dataframe(monthly_sales.round(2), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Filtered Transaction Details")

    product_search = st.text_input("Search for a product")
    detail_df = filtered.copy()

    if product_search.strip():
        detail_df = detail_df[
            detail_df["Description"].str.contains(
                product_search.strip(),
                case=False,
                na=False
            )
        ]

    st.write(f"Showing **{len(detail_df):,}** transaction rows.")

    st.dataframe(
        detail_df[[
            "InvoiceNo", "InvoiceDate", "Description",
            "Quantity", "UnitPrice", "Revenue", "Country"
        ]].sort_values("InvoiceDate", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Filtered Transactions",
        data=detail_df.to_csv(index=False).encode("utf-8"),
        file_name="spendsense_product_sales.csv",
        mime="text/csv"
    )

st.divider()

st.download_button(
    "📥 Export Product Performance Report",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="spendsense_sales_report.csv",
    mime="text/csv",
    key="export_full_sales"
)

st.caption(
    "SpendSense AI | Product analytics and retail intelligence. "
    "Revenue represents Quantity × UnitPrice for valid positive-quantity transactions."
)
