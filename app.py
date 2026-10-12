
import streamlit as st
from utils.style import apply_custom_styles

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SpendSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ai_insights = st.Page(
    "pages/06_AI_Insights.py",
    title="AI Customer Insights",
    icon="🤖"
)

forecasting = st.Page(
    "pages/07_Sales_Forecasting.py",
    title="Sales Forecasting",
    icon="📈",
)

# --------------------------------------------------
# 2. APPLY SHARED STYLING
# --------------------------------------------------
apply_custom_styles()

# --------------------------------------------------
# 3. SIDEBAR BRANDING
# --------------------------------------------------
st.sidebar.title("💳 SpendSense AI")
st.sidebar.caption("Smart Retail Intelligence")
st.sidebar.divider()

# --------------------------------------------------
# 4. DEFINE ALL APPLICATION PAGES
# --------------------------------------------------

dashboard = st.Page(
    "pages/01_Dashboard.py",
    title="Executive Dashboard",
    icon="📊",
    default=True
)

transactions = st.Page(
    "pages/02_Transaction_Explorer.py",
    title="Transaction Explorer",
    icon="🧾"
)

segmentation = st.Page(
    "pages/03_Customer_Segmentation.py",
    title="Customer Segmentation",
    icon="👥"
)

products = st.Page(
    "pages/04_Product_Insights.py",
    title="Product Insights",
    icon="📦"
)

reports = st.Page(
    "pages/05_Reports.py",
    title="Reports & Analytics",
    icon="📑"
)

model_comparison = st.Page(
    "pages/08_Model_Comparison.py",
    title="Model Comparison",
    icon="🧠",
)

# --------------------------------------------------
# 5. CREATE NAVIGATION MENU
# --------------------------------------------------

navigation = st.navigation(
    [
        dashboard,
        transactions,
        segmentation,
        products,
        reports,
        ai_insights,
        forecasting,
    ]
)


# --------------------------------------------------
# 6. RUN THE SELECTED PAGE
# --------------------------------------------------
navigation.run()
