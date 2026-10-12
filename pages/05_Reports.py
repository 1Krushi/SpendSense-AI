
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak
)
from datetime import datetime

st.set_page_config(
    page_title="Reports | SpendSense AI",
    page_icon="📑",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "online_retail.xlsx"

st.title("📑 Reports & Business Intelligence")
st.caption("Generate professional retail analytics reports and export your data.")

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
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df = df.dropna(subset=[
        "InvoiceNo", "InvoiceDate", "Description",
        "Quantity", "UnitPrice", "Country"
    ]).copy()

    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df


def create_pdf(df, title, start_date, end_date):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#17365D"),
        spaceBefore=12,
        spaceAfter=8
    ))

    story = []

    story.append(Paragraph("SpendSense AI", styles["ReportTitle"]))
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"]
    ))
    story.append(Paragraph(
        f"Period covered: {start_date} to {end_date}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 18))

    revenue = df["Revenue"].sum()
    orders = df["InvoiceNo"].nunique()
    customers = df["CustomerID"].nunique()
    units = df["Quantity"].sum()
    average_order = revenue / orders if orders else 0

    story.append(Paragraph("Executive Summary", styles["SectionHeading"]))

    summary_data = [
        ["Metric", "Value"],
        ["Total Revenue", f"GBP {revenue:,.2f}"],
        ["Unique Orders", f"{orders:,}"],
        ["Unique Customers", f"{customers:,}"],
        ["Units Sold", f"{units:,.0f}"],
        ["Average Order Value", f"GBP {average_order:,.2f}"]
    ]

    summary_table = Table(summary_data, colWidths=[2.6 * inch, 2.6 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DEE8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F1F5F9")
        ]),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.append(summary_table)

    story.append(Paragraph("Top 10 Products by Revenue", styles["SectionHeading"]))

    top_products = (
        df.groupby("Description", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Units=("Quantity", "sum")
        )
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    product_data = [["Product", "Revenue (GBP)", "Units Sold"]]
    for _, row in top_products.iterrows():
        product_name = str(row["Description"])[:65]
        product_data.append([
            Paragraph(product_name, styles["BodyText"]),
            f"{row['Revenue']:,.2f}",
            f"{row['Units']:,.0f}"
        ])

    product_table = Table(
        product_data,
        colWidths=[5.4 * inch, 1.5 * inch, 1.3 * inch],
        repeatRows=1
    )
    product_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DEE8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F1F5F9")
        ]),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))
    story.append(product_table)

    story.append(Paragraph("Monthly Revenue Summary", styles["SectionHeading"]))

    monthly = (
        df.groupby("Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique")
        )
        .sort_values("Month")
        .tail(18)
    )

    monthly_data = [["Month", "Revenue (GBP)", "Orders"]]
    for _, row in monthly.iterrows():
        monthly_data.append([
            str(row["Month"]),
            f"{row['Revenue']:,.2f}",
            f"{row['Orders']:,}"
        ])

    monthly_table = Table(
        monthly_data,
        colWidths=[2.0 * inch, 2.4 * inch, 1.5 * inch],
        repeatRows=1
    )
    monthly_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DEE8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F1F5F9")
        ]),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))
    story.append(monthly_table)

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Methodology: Revenue is calculated as Quantity × UnitPrice. "
        "This report excludes cancellation invoices, non-positive quantities "
        "and non-positive prices. Customer counts include records with an "
        "available CustomerID. Results are descriptive and do not guarantee "
        "future performance.",
        styles["Italic"]
    ))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(
            landscape(A4)[0] - 35,
            18,
            f"SpendSense AI | Page {doc.page}"
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    return buffer.getvalue()


if not DATA_FILE.exists():
    st.error("Dataset not found. Put online_retail.xlsx inside the data folder.")
    st.stop()

try:
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if df.empty:
    st.warning("No valid transactions were found in the dataset.")
    st.stop()

st.sidebar.header("Report Configuration")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Select report period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

country_options = sorted(df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select countries",
    options=country_options,
    default=country_options
)

report_title = st.sidebar.text_input(
    "Report title",
    value="Retail Sales Performance Report"
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("The start date must be on or before the end date.")
    st.stop()

filtered = df[
    df["InvoiceDate"].dt.date.between(start_date, end_date)
    & df["Country"].isin(selected_countries)
].copy()

if filtered.empty:
    st.warning("No transactions match your selected report filters.")
    st.stop()

revenue = filtered["Revenue"].sum()
orders = filtered["InvoiceNo"].nunique()
customers = filtered["CustomerID"].nunique()
units = filtered["Quantity"].sum()
aov = revenue / orders if orders else 0

st.subheader("Report Preview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"£{revenue:,.2f}")
c2.metric("Orders", f"{orders:,}")
c3.metric("Customers", f"{customers:,}")
c4.metric("Average Order Value", f"£{aov:,.2f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Monthly Revenue")
    monthly = (
        filtered.groupby("Month", as_index=False)["Revenue"]
        .sum()
        .sort_values("Month")
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Revenue",
        markers=True,
        template="plotly_white",
        labels={"Revenue": "Revenue (£)", "Month": "Month"}
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Top 10 Products")
    top_products = (
        filtered.groupby("Description", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_products.sort_values("Revenue"),
        x="Revenue",
        y="Description",
        orientation="h",
        template="plotly_white",
        labels={"Revenue": "Revenue (£)", "Description": "Product"}
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Report Actions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### PDF Business Report")
    st.write(
        "Create a formatted PDF with KPIs, top products, monthly revenue "
        "and the report period."
    )

    try:
        pdf_bytes = create_pdf(
            filtered, report_title, start_date, end_date
        )

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="spendsense_business_report.pdf",
            mime="application/pdf"
        )
    except Exception as error:
        st.error(f"Could not generate the PDF: {error}")

with col2:
    st.markdown("#### CSV Data Export")
    st.write("Export the transactions matching the report filters.")

    st.download_button(
        "📥 Download Filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="spendsense_report_data.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "SpendSense AI | Retail analytics reporting. "
    "Figures describe the selected historical dataset."
)
