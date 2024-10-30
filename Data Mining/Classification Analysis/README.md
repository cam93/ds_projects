# Customer Churn Prediction Using K-Nearest Neighbor (KNN) Classification

This project investigates customer churn for a telecommunications company using the k-Nearest Neighbor (KNN) classification algorithm. The analysis aims to identify key customer features that indicate churn likelihood, enabling the company to develop proactive retention strategies.

## Project Overview

### Research Question
**"Which specific features most strongly indicate whether a customer is likely to churn?"**

### Goals
1. **Predict Churn Risk**: Develop a KNN model to classify customers based on churn likelihood.
2. **Identify Key Churn Drivers**: Analyze customer features (e.g., monthly charges, internet service type, tech support) that are most indicative of churn.
3. **Implement Balanced Classification**: Address dataset imbalance to improve the model's ability to detect at-risk customers accurately.

## Data and Methodology

### Data Preprocessing
- **Handling Categorical Variables**: Categorical data such as `internet_service` was transformed into a binary variable, simplifying `DSL` and `Fiber Optic` to 1 (internet available) and `N/A` to 0.
- **Binary Encoding**: Categorical columns (e.g., `techie`, `online_security`) were encoded to numeric values for compatibility with KNN.
- **Dropped Irrelevant Columns**: Features not contributing to churn prediction were removed, streamlining the model’s focus.

### K-Nearest Neighbor (KNN) Model
1. **Distance Calculation**: Used Euclidean distance to determine proximity in feature space.
2. **Optimal K Selection**: Tested multiple `k` values using GridSearchCV to find the optimal number of neighbors.
3. **Training and Testing**: Applied an 80/20 train-test split to evaluate model performance.

### Evaluation Metrics
- **Accuracy**: Achieved 74.9%, indicating the model’s overall performance.
- **Confusion Matrix**: Showed high false negatives, signaling a need for improved churn detection.
- **AUC Score**: Scored 0.725, reflecting moderate model ability to distinguish between churned and non-churned customers.

## Results Summary

The KNN model’s results indicate a reasonable accuracy level; however, there is room for improvement, especially in correctly identifying churned customers.

### Key Findings
- **High False Negatives**: The model struggles to identify churned customers accurately, impacting retention strategy effectiveness.
- **AUC Score**: At 0.725, the model has moderate classification power but requires enhancements for improved discrimination.

### Recommendations
1. **Enhance Model with Imbalance Techniques**: Address dataset imbalance with oversampling, undersampling, or synthetic methods (e.g., SMOTE).
2. **Explore Alternative Models**: Consider models like Random Forest or XGBoost for better handling of complex data patterns.
3. **Implement Targeted Retention**: Focus retention efforts on customers with high churn risk, particularly those lacking tech support or online security.

## Technical Requirements

### Libraries
- **pandas** and **NumPy**: for data handling and manipulation.
- **scikit-learn**: for KNN model implementation, hyperparameter tuning, and evaluation.
- **matplotlib** and **seaborn**: for data visualization and exploratory analysis.

## File Descriptions

- **customer_churnPrediction.pdf**: Documentation of research question, data preprocessing, and methodology.
- **knn_classification.ipynb**: Jupyter notebook containing code for data cleaning, KNN implementation, and evaluation.