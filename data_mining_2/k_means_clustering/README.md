# Customer Segmentation Using K-Means Clustering

This project applies K-means clustering to segment customers based on demographics and service usage patterns. By understanding these segments, the organization can create tailored marketing strategies, optimize resources, and improve customer retention and satisfaction.

## Project Overview

### Objective
The main question addressed is:
**"How can we segment customers based on demographics and service usage to identify groups for targeted marketing?"**

### Goals
1. **Customer Segmentation**: Group customers by demographic and usage patterns to reveal actionable insights.
2. **Targeted Marketing**: Develop marketing strategies suited to each segment, aiming to boost retention and satisfaction.

## Data and Methodology

### Key Data Variables
- **Age**: Customer's age.
- **Income**: Annual income.
- **MonthlyCharge**: Monthly service charge.
- **Bandwidth_GB_Year**: Yearly bandwidth usage.

### Analysis Steps
1. **Data Preparation**: Checked for missing values, renamed columns, and scaled data to equalize feature influence.
2. **Optimal Cluster Determination**: Used the Elbow method to select the optimal number of clusters (k=7).
3. **K-means Clustering**: Clustered customers and calculated centroids for each group to interpret segment characteristics.
4. **Cluster Evaluation**: Used metrics like inertia, silhouette score, and Davies-Bouldin score to evaluate cluster quality.

### Libraries
- **pandas** and **NumPy**: Data handling.
- **scikit-learn**: K-means clustering, evaluation metrics, scaling.
- **matplotlib** and **seaborn**: Data visualization.

## Key Insights and Results

### Clusters Identified
Each cluster provides insights into distinct customer groups. For instance:
- **High-income, low-bandwidth users**: Ideal for premium services.
- **Young, low-income, high-bandwidth users**: Suited for affordable, high-data plans.
- **Moderate-income, high-charge customers**: Retention focus with tailored value-added packages.

### Evaluation
- **Silhouette Score**: 0.251, indicating moderate clustering quality.
- **Davies-Bouldin Score**: 1.165, showing reasonable cluster separation.

## Limitations
- **Cluster Shape Assumption**: K-means assumes spherical clusters, which may not fit real-world customer data patterns. Future analysis may consider advanced techniques like Gaussian Mixture Models.

## Recommended Actions
1. **Cluster-Specific Marketing**: Design campaigns tailored to each group.
2. **Personalized Retention**: Use cluster insights to address specific customer needs and reduce churn.
3. **Service Optimization**: Allocate resources based on segment demand.
4. **Continuous Reassessment**: Regularly update clustering to keep insights relevant.

## Files and Dependencies

- **k_means.ipynb**: Jupyter notebook containing data preprocessing, clustering, and evaluation code.
- **analysis_report.pdf**: Documentation of the project, including methodology, clustering results, and recommended actions.