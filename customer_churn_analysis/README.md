# Customer Churn Analysis

This project analyzes customer churn for a telecom provider to identify key factors contributing to churn, improve service, and devise targeted retention strategies. By analyzing data on service usage, customer demographics, and satisfaction metrics, the aim is to provide actionable insights for reducing churn and enhancing customer loyalty.

## Project Structure

- **data_cleaning.ipynb**: Jupyter notebook for data cleaning, including handling missing values, detecting outliers, and renaming columns for clarity.
- **cleaned_data.csv**: Processed dataset post-cleaning, ready for analysis.
  
## Project Details

### 1. Objective
Identify the primary factors influencing customer churn in order to mitigate churn rates, maintain a strong customer base, and sustain a competitive edge in the telecom market.

### 2. Data
The dataset includes:
- **Customer demographics**: age, income, employment status, marital status, etc.
- **Service usage**: Internet, phone, streaming services, security add-ons, etc.
- **Customer satisfaction**: Survey responses on service timeliness, reliability, and customer service.

### 3. Data Cleaning
Key steps for data cleaning:
- **Handling missing values**: Imputed with mean/median values for numerical fields and mode for categorical fields.
- **Outlier detection and mitigation**: Identified using Z-scores and box plots, with special attention to `income` and `population`.
- **Standardization**: Converted time zones to U.S. standards and used snake_case naming conventions.
  
### 4. Analysis Methods
- **Principal Component Analysis (PCA)**: Reduced dimensionality to focus on key factors like service usage and customer demographics.
- **Regression Analysis**: Explored correlations between variables like monthly charges, contract type, and churn likelihood.

### 5. Limitations
- **Imputation of missing values** may introduce bias.
- **Standardized U.S. time zones**, which might omit global insights.
- **Categorical conversion** limits geographical analysis of certain fields.

### 6. Outcomes
Identified critical churn factors:
- Timeliness of responses and service repairs
- Contract length and monthly charges
- Availability of additional services (e.g., online security, tech support)

## Requirements
This project is developed in Python using the following libraries:
- **pandas**: for data manipulation
- **numpy**: for numerical operations
- **scipy**: for statistical computations
- **matplotlib** and **seaborn**: for data visualization
- **scikit-learn**: for dimensionality reduction and machine learning.