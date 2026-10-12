
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.style import apply_custom_styles


# ============================================================
# 1. PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="SpendSense AI | Executive Dashboard",
    page_icon="📊",
    layout="wide",
)

apply_custom_styles()

st.title("📊 Executive Dashboard")
st.caption(
    "SpendSense AI | Retail performance, customer intelligence "
    "and data-driven business insights"
)

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "online_retail.xlsx"
)


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_excel(file_path)

    required_columns = [
        "InvoiceNo",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
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
        ]
    )

    # Exclude cancelled invoices and invalid sales records.
    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ]

    return df


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info(
        "Make sure online_retail.xlsx is saved inside "
        "the project's data folder."
    )
    st.stop()

try:
    df = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if df.empty:
    st.warning("The dataset contains no valid sales records.")
    st.stop()


# ============================================================
# 3. FILTERS
# ============================================================

st.sidebar.subheader("🎛️ Dashboard Filters")

minimum_date = df["InvoiceDate"].min().date()
maximum_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Sales date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)

all_countries = sorted(
    df["Country"].dropna().unique().tolist()
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=all_countries,
    default=all_countries,
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.warning("The start date must be before the end date.")
    st.stop()

if not selected_countries:
    st.warning("Select at least one country to view analytics.")
    st.stop()

filtered = df[
    (df["InvoiceDate"].dt.date >= start_date)
    & (df["InvoiceDate"].dt.date <= end_date)
    & (df["Country"].isin(selected_countries))
].copy()

if filtered.empty:
    st.warning(
        "No sales records match these filters. "
        "Try another date range or country."
    )
    st.stop()

st.caption(
    f"Analysis period: {start_date:%d %b %Y} – "
    f"{end_date:%d %b %Y} | "
    f"{len(selected_countries)} countries selected"
)


# ============================================================
# 4. METRIC HELPER FUNCTIONS
# ============================================================

def calculate_metrics(data):
    orders = data["InvoiceNo"].nunique()

    customers = data.loc[
        data["CustomerID"].notna()
        & (data["CustomerID"].str.strip() != ""),
        "CustomerID",
    ].nunique()

    return {
        "revenue": data["Revenue"].sum(),
        "orders": orders,
        "customers": customers,
        "units": data["Quantity"].sum(),
        "aov": (
            data["Revenue"].sum() / orders
            if orders else 0
        ),
    }


def growth_percentage(current, previous):
    if previous > 0:
        return (current - previous) / previous * 100

    return None


def growth_label(current, previous):
    growth = growth_percentage(current, previous)

    if growth is None:
        return "No previous-period baseline"

    return f"{growth:+.1f}% vs previous period"


current = calculate_metrics(filtered)


# ============================================================
# 5. PREVIOUS-PERIOD COMPARISON
# ============================================================

period_days = (end_date - start_date).days + 1

previous_end = pd.Timestamp(start_date) - pd.Timedelta(days=1)
previous_start = previous_end - pd.Timedelta(
    days=period_days - 1
)

previous = df[
    (df["InvoiceDate"].dt.date >= previous_start.date())
    & (df["InvoiceDate"].dt.date <= previous_end.date())
    & (df["Country"].isin(selected_countries))
].copy()

previous_metrics = calculate_metrics(previous)


# ============================================================
# 6. KPI CARDS
# ============================================================

st.subheader("Business Performance")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Revenue",
    f"£{current['revenue']:,.2f}",
    growth_label(
        current["revenue"], previous_metrics["revenue"]
    ),
)

k2.metric(
    "Total Orders",
    f"{current['orders']:,}",
    growth_label(
        current["orders"], previous_metrics["orders"]
    ),
)

k3.metric(
    "Unique Customers",
    f"{current['customers']:,}",
)

k4.metric(
    "Units Sold",
    f"{current['units']:,.0f}",
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Average Order Value",
    f"£{current['aov']:,.2f}",
)

m2.metric(
    "Revenue per Customer",
    (
        f"£{current['revenue'] / current['customers']:,.2f}"
        if current["customers"] else "N/A"
    ),
)

m3.metric(
    "Revenue Change",
    growth_label(
        current["revenue"], previous_metrics["revenue"]
    ),
)

st.caption(
    "Currency: GBP (£), as used by the original Online Retail dataset. "
    "Growth compares the selected period with the immediately "
    "preceding period of equal length. A positive change is not "
    "necessarily evidence of improved profitability."
)

st.divider()


# ============================================================
# 7. REVENUE AND ORDER TRENDS
# ============================================================

st.subheader("📈 Sales Performance Trends")

trend_data = filtered.copy()

trend_data["Month"] = (
    trend_data["InvoiceDate"].dt.to_period("M").astype(str)
)

monthly = (
    trend_data.groupby("Month", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
    )
    .sort_values("Month")
)

left, right = st.columns(2)

with left:
    revenue_chart = px.area(
        monthly,
        x="Month",
        y="Revenue",
        markers=True,
        title="Monthly Revenue",
        labels={
            "Month": "Month",
            "Revenue": "Revenue (£)",
        },
        template="plotly_white",
    )

    revenue_chart.update_traces(
        line=dict(color="#2563eb", width=3),
        fillcolor="rgba(37, 99, 235, 0.12)",
    )

    revenue_chart.update_layout(
        height=370,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )

    st.plotly_chart(
        revenue_chart,
        use_container_width=True,
    )

with right:
    order_chart = px.bar(
        monthly,
        x="Month",
        y="Orders",
        title="Monthly Order Volume",
        labels={
            "Month": "Month",
            "Orders": "Unique Orders",
        },
        template="plotly_white",
    )

    order_chart.update_traces(
        marker_color="#0d9488"
    )

    order_chart.update_layout(
        height=370,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(
        order_chart,
        use_container_width=True,
    )


# ============================================================
# 8. PRODUCT AND COUNTRY PERFORMANCE
# ============================================================

st.divider()
st.subheader("🏆 Product and Market Intelligence")

left, right = st.columns(2)

with left:
    product_sales = (
        filtered.groupby("Description", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Units=("Quantity", "sum"),
        )
        .sort_values("Revenue", ascending=False)
        .head(10)
        .sort_values("Revenue")
    )

    product_chart = px.bar(
        product_sales,
        x="Revenue",
        y="Description",
        orientation="h",
        title="Top 10 Products by Revenue",
        labels={
            "Revenue": "Revenue (£)",
            "Description": "Product",
        },
        template="plotly_white",
    )

    product_chart.update_traces(
        marker_color="#6366f1"
    )

    product_chart.update_layout(
        height=450,
        yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(
        product_chart,
        use_container_width=True,
    )

with right:
    country_sales = (
        filtered.groupby("Country", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(10)
        .sort_values("Revenue")
    )

    country_chart = px.bar(
        country_sales,
        x="Revenue",
        y="Country",
        orientation="h",
        title="Top 10 Countries by Revenue",
        labels={
            "Revenue": "Revenue (£)",
            "Country": "Country",
        },
        template="plotly_white",
    )

    country_chart.update_traces(
        marker_color="#0d9488"
    )

    country_chart.update_layout(
        height=450,
        yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(
        country_chart,
        use_container_width=True,
    )


# ============================================================
# 9. CUSTOMER ANALYTICS
# ============================================================

st.divider()
st.subheader("👥 Customer Intelligence")

identified = filtered[
    filtered["CustomerID"].notna()
    & (filtered["CustomerID"].str.strip() != "")
].copy()

if identified.empty:
    st.info(
        "Customer-level analysis is unavailable because "
        "the selected records have no customer IDs."
    )
else:
    customer_orders = (
        identified.groupby("CustomerID")
        .agg(
            Orders=("InvoiceNo", "nunique"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    repeat_customers = int(
        (customer_orders["Orders"] >= 2).sum()
    )

    total_identified = len(customer_orders)

    repeat_rate = (
        repeat_customers / total_identified * 100
        if total_identified else 0
    )

    top_customers = customer_orders.nlargest(
        10, "Revenue"
    ).copy()

    a, b = st.columns([1, 2])

    with a:
        st.metric(
            "Repeat Customer Rate",
            f"{repeat_rate:.1f}%",
        )
        st.caption(
            "Share of identified customers with two or more "
            "distinct invoices in the selected period."
        )

        st.metric(
            "Identified Customers",
            f"{total_identified:,}",
        )

    with b:
        customer_chart = px.bar(
            top_customers.sort_values("Revenue"),
            x="Revenue",
            y="CustomerID",
            orientation="h",
            title="Top 10 Customers by Revenue",
            labels={
                "Revenue": "Revenue (£)",
                "CustomerID": "Customer ID",
            },
            template="plotly_white",
        )

        customer_chart.update_layout(
            height=390,
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=10),
        )

        st.plotly_chart(
            customer_chart,
            use_container_width=True,
        )


# ============================================================
# 10. AUTOMATIC BUSINESS INSIGHTS
# ============================================================

st.divider()
st.subheader("💡 Automatic Business Insights")

insights = []

# Revenue insight
if len(monthly) >= 2:
    latest_revenue = monthly.iloc[-1]["Revenue"]
    previous_month_revenue = monthly.iloc[-2]["Revenue"]

    if previous_month_revenue > 0:
        monthly_change = (
            (latest_revenue - previous_month_revenue)
            / previous_month_revenue
        ) * 100

        if monthly_change > 0:
            insights.append(
                f"📈 Monthly revenue increased by "
                f"{monthly_change:.1f}% in {monthly.iloc[-1]['Month']} "
                f"compared with {monthly.iloc[-2]['Month']}."
            )
        elif monthly_change < 0:
            insights.append(
                f"📉 Monthly revenue decreased by "
                f"{abs(monthly_change):.1f}% in {monthly.iloc[-1]['Month']} "
                f"compared with {monthly.iloc[-2]['Month']}. "
                "Review product and market performance for possible causes."
            )
        else:
            insights.append(
                "➡️ Revenue remained unchanged between the two "
                "latest months in the selected data."
            )
    else:
        insights.append(
            "ℹ️ There is no positive previous-month revenue baseline "
            "for a reliable monthly growth comparison."
        )
else:
    insights.append(
        "ℹ️ Select a period containing at least two months "
        "to compare monthly revenue trends."
    )

# Product insight
if not product_sales.empty:
    best_product = product_sales.sort_values(
        "Revenue", ascending=False
    ).iloc[0]

    insights.append(
        f"🏆 Top product by revenue: "
        f"{best_product['Description']} "
        f"(£{best_product['Revenue']:,.2f})."
    )

# Country insight
if not country_sales.empty and current["revenue"] > 0:
    best_country = country_sales.sort_values(
        "Revenue", ascending=False
    ).iloc[0]

    country_share = (
        best_country["Revenue"] / current["revenue"]
    ) * 100

    insights.append(
        f"🌍 {best_country['Country']} is the leading country "
        f"by revenue, contributing {country_share:.1f}% "
        "of revenue in the selected data."
    )

# Customer insight
if not identified.empty:
    if repeat_rate >= 30:
        insights.append(
            f"👥 {repeat_rate:.1f}% of identified customers "
            "placed multiple orders in this period. "
            "Consider loyalty offers to encourage continued engagement."
        )
    else:
        insights.append(
            f"👥 The repeat-customer rate is {repeat_rate:.1f}%. "
            "Consider testing retention campaigns and measuring "
            "their results over time."
        )

# Average order value insight
if current["aov"] > 0:
    insights.append(
        f"🛒 Average order value is £{current['aov']:,.2f}. "
        "Compare this metric across periods before evaluating "
        "bundling or cross-selling initiatives."
    )

for insight in insights:
    st.info(insight)

st.caption(
    "Insights are rule-based summaries calculated from the selected data. "
    "They identify patterns, not proven causes or guaranteed future outcomes."
)


# ============================================================
# 11. TOP PRODUCTS TABLE
# ============================================================

st.divider()
st.subheader("📦 Product Performance Table")

if not product_sales.empty:
    table = (
        filtered.groupby("Description", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Units_Sold=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    table.index = table.index + 1

    st.dataframe(
        table,
        use_container_width=True,
    )


# ============================================================
# 12. RECENT TRANSACTIONS AND EXPORT
# ============================================================

st.divider()
st.subheader("🧾 Recent Transactions")

display_columns = [
    "InvoiceNo",
    "InvoiceDate",
    "Description",
    "Quantity",
    "UnitPrice",
    "Revenue",
    "Country",
]

recent = (
    filtered.sort_values("InvoiceDate", ascending=False)
    [display_columns]
    .head(20)
    .copy()
)

recent["InvoiceDate"] = recent["InvoiceDate"].dt.strftime(
    "%Y-%m-%d %H:%M"
)

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Sales Data",
    data=csv_data,
    file_name="spendsense_filtered_sales.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "SpendSense AI | Retail intelligence powered by data analytics"
)
