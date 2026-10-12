# SpendSense AI — Smart Retail Intelligence

SpendSense AI is an interactive retail analytics application that transforms online retail transaction data into useful business insights. It helps users explore sales performance, understand customer purchasing behavior, identify popular products, and examine potential sales trends using data analytics and machine learning.

## Features

- **Executive Dashboard:** View key performance indicators, revenue trends, and business summaries.
- **Transaction Explorer:** Filter and investigate retail transactions.
- **Customer Segmentation:** Group customers based on purchasing behavior.
- **Product Insights:** Identify popular products and analyze product revenue.
- **Reports & Analytics:** Explore summarized results and export data.
- **AI Customer Insights:** Generate data-driven observations to support business decisions.
- **Sales Forecasting:** Estimate future daily revenue using historical transaction data.
- **Model Comparison:** Compare Linear Regression, Decision Tree, and Random Forest models using evaluation metrics.
- **Interactive Visualizations:** Explore data through charts, filters, and tables.
- **CSV Exports:** Download selected results for further analysis.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Scikit-learn
- Joblib
- OpenPyXL
- ReportLab (if used by the reporting module)

## Dataset

The project uses the Online Retail dataset, which contains transaction records from an online retailer.

Place the dataset in this location:

`data/online_retail.xlsx`

The application analyzes transaction dates, product quantities, unit prices, invoice details, and customer information when available.

## Project Structure

```text
spendsense/
├── app.py
├── requirements.txt
├── README.md
├── pages/
│   ├── 01_Dashboard.py
│   ├── 02_Transaction_Explorer.py
│   ├── 03_Customer_Segmentation.py
│   ├── 04_Product_Insights.py
│   ├── 05_Reports.py
│   ├── 06_AI_Insights.py
│   ├── 07_Sales_Forecasting.py
│   └── 08_Model_Comparison.py
├── data/
│   └── online_retail.xlsx
├── models/
├── utils/
└── assets/
    └── style.css
```

## Installation and Setup

### 1. Install Python

Install a compatible Python version and verify the installation:

```bash
python --version
```

### 2. Open the project directory

```bash
cd C:\Users\krush\spendsense
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Add the dataset

Ensure the dataset is available at `data/online_retail.xlsx`.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Evaluation Metrics

The forecasting model comparison includes:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted revenue.
- **RMSE (Root Mean Squared Error):** Error metric that penalizes larger prediction errors more strongly.
- **R² (Coefficient of Determination):** Indicates how well a model explains variation in the evaluation data.

Model performance depends on the dataset, feature design, and evaluation period. Forecasts are estimates, not guaranteed future sales.

## Business Benefits

- Understand revenue patterns and sales performance.
- Identify products and customers that contribute to business activity.
- Support inventory and sales planning.
- Explore purchasing trends using interactive analytics.
- Compare predictive models before choosing a forecasting approach.

## Limitations

- Forecast quality depends on historical data and model assumptions.
- Revenue is not the same as profit because operating costs and product costs are not included.
- The project is an analytics prototype and does not automatically execute business decisions.
- Results should be reviewed before being used for real-world planning.

## Future Enhancements

- Live database integration.
- Automated data refresh.
- Advanced customer lifetime value analysis.
- Improved forecasting with time-series models.
- User authentication and role-based access.
- Cloud deployment and scheduled reports.

## Project Status

An educational retail intelligence prototype combining interactive analytics with machine-learning experiments. Features should be tested against the available dataset before release.
