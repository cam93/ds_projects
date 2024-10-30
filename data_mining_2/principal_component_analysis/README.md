# Customer Behavior Analysis Using Principal Component Analysis (PCA)

This project utilizes Principal Component Analysis (PCA) to reduce the dimensionality of a customer dataset while retaining critical features related to customer behavior. The analysis aims to highlight the most influential components associated with metrics such as tenure, monthly charges, and service usage patterns. By reducing the number of variables, PCA enhances interpretability and computational efficiency, facilitating deeper insights into customer behavior.

## Project Overview

### Objective
The main question this analysis seeks to answer is:
**"Can Principal Component Analysis (PCA) reduce dimensionality while identifying key components related to customer behavior?"**

### Goals
1. **Dimensionality Reduction**: Summarize the dataset with a smaller number of components that capture the majority of variability.
2. **Customer Insight**: Highlight components that reflect underlying patterns in customer tenure, charges, and usage.
3. **Improved Computational Efficiency**: Enable faster processing and streamlined visualization by reducing variables.

## Data and Methodology

### Key Variables
The continuous variables analyzed are:
- **tenure**: Customer tenure duration.
- **monthly_charge**: Monthly charges incurred.
- **outage_sec_perweek**: Weekly service outage duration in seconds.
- **bandwidth_gb_year**: Yearly bandwidth usage.

### PCA Analysis Steps
1. **Standardization**: Centered and scaled variables to ensure equal influence.
2. **Covariance and Eigen Decomposition**: Calculated the covariance matrix, eigenvalues, and eigenvectors.
3. **Principal Component Selection**: Used the Kaiser criterion, retaining components with eigenvalues > 1, which identified two principal components.
4. **Variance Explained**: Calculated cumulative variance for the selected components, achieving 75.34% explained variance with two components.

### Libraries Used
- **pandas** and **NumPy**: Data handling.
- **scikit-learn**: PCA, scaling.
- **matplotlib**: Data visualization.

## Key Results

### Principal Components
- **PC1**: Explained 49.83% of variance, highlighting core patterns in customer metrics.
- **PC2**: Explained 25.51% of variance, capturing additional unique customer behavior insights.

### Cumulative Variance
- The first two components capture 75.34% of total variance, making them sufficient to summarize the dataset’s core features.

## Benefits and Insights
1. **Data Visualization**: Simplified to two dimensions, revealing customer clusters and trends.
2. **Efficiency in Computation**: Reduced dimensionality enhances model training speed and reduces resource usage.
3. **Feature Focused Models**: Models built on these components are less complex and more generalizable.

## Limitations
- **Linearity Assumption**: PCA assumes linear relationships, potentially missing nonlinear patterns in customer data.

## Files and Dependencies

- **pca_analysis.ipynb**: Notebook containing PCA code, from data standardization through to analysis of principal components.
- **pca.pdf**: Documentation detailing project methodology, findings, and recommendations.