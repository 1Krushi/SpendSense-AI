
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Transaction Explorer | SpendSense AI",
    page_icon="🧾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "online_retail.xlsx"

st.title("Transaction Explorer")
st.caption("Search, filter and investigate retail transactions.")

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["CustomerID", "Description"]).copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df

if not DATA_FILE.exists():
    st.error("Dataset not found. Place online_retail.xlsx in the data folder.")
    st.stop()

df = load_data(DATA_FILE)

# Sidebar filters
st.sidebar.header("Transaction Filters")

countries = sorted(df["Country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select countries",
    options=countries,
    default=countries[:5]
)

search = st.sidebar.text_input("Search product description")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Invoice date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
filtered = df.copy()

if selected_countries:
    filtered = filtered[filtered["Country"].isin(selected_countries)]
else:
    filtered = filtered.iloc[0:0]

if search.strip():
    filtered = filtered[
        filtered["Description"].str.contains(
            search.strip(), case=False, na=False
        )
    ]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        filtered["InvoiceDate"].dt.date.between(start_date, end_date)
    ]

# Summary cards
col1, col2, col3 = st.columns(3)

col1.metric("Matching Records", f"{len(filtered):,}")
col2.metric("Unique Invoices", f"{filtered['InvoiceNo'].nunique():,}")
col3.metric("Filtered Revenue", f"£{filtered['Revenue'].sum():,.2f}")

st.divider()

st.subheader("Transaction Records")

st.dataframe(
    filtered[
        [
            "InvoiceNo", "InvoiceDate", "CustomerID",
            "Description", "Quantity", "UnitPrice",
            "Revenue", "Country"
        ]
    ].sort_values("InvoiceDate", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Filtered Transactions",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="spendsense_filtered_transactions.csv",
    mime="text/csv"
)

st.divider()
st.subheader("Revenue by Country")

if not filtered.empty:
    country_revenue = (
        filtered.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
    )

    fig = px.bar(
        country_revenue.head(10),
        x="Country",
        y="Revenue",
        template="plotly_white",
        title="Top Countries by Filtered Revenue",
        labels={"Revenue": "Revenue (£)"}
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No transactions match your filters. Try changing the country or date range.")
