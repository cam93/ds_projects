# Customer Tenure Analysis Using Decision Tree Regression

This project applies decision tree regression to analyze factors contributing to customer tenure for a telecommunications company. The analysis identifies critical features influencing how long customers stay, providing actionable insights for enhancing customer retention and reducing churn.

## Project Overview

### Research Question
**"What factors contribute most to customer tenure?"**

### Goals
1. **Identify Key Tenure Drivers**: Determine the attributes (e.g., monthly charges, age, online security) that significantly impact customer tenure.
2. **Develop Predictive Model**: Use decision tree regression to predict tenure and highlight factors driving customer loyalty.
3. **Enhance Customer Retention**: Leverage insights to create targeted strategies that extend customer tenure and reduce churn.

## Data and Methodology

### Data Preprocessing
- **Feature Selection**: Included features like `children`, `age`, `monthly_charge`, `bandwidth_gb_year`, and `online_security`.
- **Encoding Categorical Variables**: Transformed categorical columns, such as `online_security`, into binary numeric formats.
- **Train-Test Split**: Divided data into 80% training and 20% test sets for model validation.

### Decision Tree Regression Model
1. **Model Training**: Used `DecisionTreeRegressor` from scikit-learn to model customer tenure.
2. **Hyperparameter Tuning**: Employed `GridSearchCV` to optimize `max_depth` and `min_samples_leaf` for improved accuracy.
3. **Evaluation**: Evaluated model performance using Mean Squared Error (MSE) and R-squared (R²) metrics.

### Evaluation Metrics
- **MSE**: Achieved a low MSE of 7.700, indicating minimal error in tenure prediction.
- **R-squared (R²)**: Obtained a high R² of 0.989, demonstrating the model's strong predictive power.

## Results Summary

The decision tree model demonstrates high accuracy in predicting customer tenure, with key findings that inform retention strategies.

### Key Findings
- **High Impact Features**: Variables such as `monthly_charge`, `age`, and `online_security` play significant roles in customer tenure.
- **Predictive Accuracy**: Low MSE and high R² confirm the model’s reliability in forecasting tenure duration.

### Recommendations
1. **Targeted Retention Strategies**: Offer incentives to customers with shorter predicted tenure, focusing on features like security and service upgrades.
2. **Proactive Engagement**: Implement early intervention for customers likely to leave soon, reducing potential churn.
3. **Resource Optimization**: Allocate resources effectively by focusing on high-impact retention activities driven by key tenure factors.

## Technical Requirements

### Libraries
- **pandas** and **NumPy**: for data manipulation.
- **scikit-learn**: for model implementation, tuning, and evaluation.
- **matplotlib** and **seaborn**: for data visualization.

## File Descriptions

- **tenure_analysis.pdf**: Documentation covering the project’s objectives, data preparation, methodology, and implications.
- **decision_trees.ipynb**: Jupyter notebook containing code for data cleaning, decision tree modeling, and evaluation.
