# Customer Churn Prediction Using Logistic Regression

This project analyzes customer churn using logistic regression to identify key factors influencing churn in a telecom dataset. By understanding these factors, the company can develop targeted strategies to enhance retention, reduce churn rates, and drive long-term customer loyalty.

## Project Overview

### Objective
The primary research question is:
**"What factors contribute most to customer churn?"**

### Goals
1. **Identify Churn Drivers**: Determine the variables that significantly increase churn likelihood, including service usage, customer demographics, and behavioral attributes.
2. **Develop Predictive Model**: Build a logistic regression model to forecast churn risk, enabling proactive retention efforts for at-risk customers.

## Data and Methodology

### Key Variables
- **Dependent Variable**: Churn (binary indicator of whether a customer churned or not)
- **Independent Variables**: Include tech-savviness, phone and internet service features, contract length, streaming services, and additional security and tech support options.

### Analytical Approach
1. **Data Cleaning**: Performed categorical encoding, handled missing values, removed irrelevant columns, and addressed multicollinearity.
2. **Logistic Regression Modeling**: Built and refined the model with backward feature selection to retain only statistically significant predictors.
3. **Evaluation**: Compared the initial and reduced models using pseudo R-squared and classification accuracy to ensure interpretability and predictive accuracy.

## Results Summary

- **Initial vs. Reduced Model**: 
  - The initial model included 16 variables with a pseudo R-squared of 0.1890.
  - The reduced model focused on 11 key variables and achieved a similar fit with a pseudo R-squared of 0.1887, retaining only statistically significant variables.
  
- **Insights**:
  - **High churn risk** among customers with multiple lines, streaming services, and those without security features.
  - **Reduced churn** associated with one-year contracts and phone service subscriptions.

### Recommendations

1. **Targeted Retention for High-Risk Groups**: Offer special incentives for customers with multiple lines and streaming services.
2. **Enhance Security Offerings**: Promote online security services to lower churn rates.
3. **Contract Commitment Offers**: Encourage customers to select one-year contracts through targeted promotions.
4. **Retain Tech-Savvy Customers**: Provide advanced features and early access to attract and retain tech-savvy customers.

## Technical Details

### Requirements
This project uses Python with the following libraries:
- **pandas**: for data manipulation
- **scikit-learn**: for logistic regression and model evaluation
- **statsmodels**: for statistical analysis and feature selection
- **matplotlib** and **seaborn**: for data visualization