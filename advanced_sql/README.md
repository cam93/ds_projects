# Customer Churn Dashboard Analysis

This project uses interactive dashboards in Tableau to analyze and visualize customer churn data for a telecommunications provider, integrating internal and external datasets. By examining factors such as contract type, payment methods, tenure, and customer demographics, the dashboard supports data-driven strategies for improving customer retention.

## Project Overview

### Objective
The primary goal is to provide stakeholders with insights into customer churn patterns, focusing on variables that may influence retention. By comparing internal data with competitor data, the dashboard helps identify key trends and areas for improvement.

### Datasets
1. **Internal Churn Data**: Customer churn data from WGU’s `churn_clean.csv`.
2. **External Data**: Kaggle’s customer churn dataset for comparative analysis ([link](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction)).

## Installation

### Setup Instructions
1. **Database Setup**:
   - Place all files (e.g., `churn_clean.csv`, SQL setup file) into `C:\Users\Public\Documents`.
   - Open `pgAdmin`, connect to the database, and execute `data_cleaning.odt` to import and clean data.
2. **Tableau Installation**:
   - Install Tableau Desktop from [Tableau’s website](https://www.tableau.com/).
   - Import `d211.twbx` Tableau file to access the dashboard.

3. **Login**: When prompted, use the following credentials:
   - **Username**: postgres
   - **Password**: Passw0rd!

## Dashboard Navigation

1. **Open Dashboard**: Access the Tableau dashboard.
2. **Explore Visuals**:
   - **Gender Distribution**: Pie chart showing customer distribution by gender.
   - **Tenure Analysis**: Bar charts to explore tenure trends by churn and gender.
   - **Payment Methods**: Bar chart detailing average charges by payment type.
3. **Interactivity**:
   - Click visuals to filter data across the dashboard.
   - Hover over elements for detailed tooltips.
   - Reset filters as needed to view the full dataset.

## Key Insights and Analysis

### Insights from the Dashboard
1. **Churn and Tenure**: Customers with shorter tenures are more likely to churn, emphasizing the need for early engagement strategies.
2. **Payment Method Insights**: Higher charges are associated with electronic check payments, suggesting a potential risk for churn that could be mitigated by promoting auto-payment options.
3. **Customer Demographics**: Gender has minimal impact on tenure, but contract types and payment preferences offer actionable retention strategies.

### Limitations
- **Lack of Behavioral Data**: The dataset lacks detailed behavioral metrics like customer engagement or satisfaction.
- **Static Snapshot**: This analysis is based on static data, limiting insights into evolving trends over time.

## Files and Dependencies

- **analysis.twbx**: Tableau workbook file for interactive data exploration.
- **data_cleaning.odt**: SQL setup file to clean and prepare data for analysis.
- **WA_Fn-UseC_-Telco-Customer-Churn.csv**: Primary churn dataset.