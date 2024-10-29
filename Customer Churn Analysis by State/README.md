# Customer Churn Analysis by State

This project examines customer churn rates across different states to determine if geographic location influences churn. It uses statistical tests to assess whether location-based patterns exist, guiding potential region-specific customer retention strategies.

## Project Overview

### Objective
The central question of this analysis is:
**"Is there a significant difference in customer churn rates between different states?"**

### Benefit
Understanding churn variation by state could help stakeholders, such as marketing and customer service teams, focus retention efforts on states with higher churn rates.

## Data and Methodology

### Key Data Columns
- **State**: Identifies each customer’s location and is used to group churn rates by state.
- **Churn**: A binary variable (Yes/No) indicating if a customer has left the service, serving as the dependent variable.

### Analytical Approach
1. **Statistical Test**: Chi-Square test to determine if churn rates vary significantly between states.
2. **Visualization**: 
   - Scatter plot for continuous variable relationship (`MonthlyCharge` vs `Bandwidth_GB_Year`)
   - Stacked bar chart to observe churn rates across states.

## Results Summary

- **Chi-Square Statistic**: 51.19
- **P-Value**: 0.99999

### Interpretation
The high p-value indicates no statistically significant difference in churn rates across states. This suggests geographic location does not significantly affect customer churn, and region-based retention strategies may not be effective.

## Limitations

- **Confounding Variables**: Other factors like service quality and pricing were not analyzed but could affect churn.
- **Chi-Square Limitations**: This test cannot measure the strength or direction of the relationship and may require additional analyses to confirm findings.
- **Assumption Dependence**: Results rely on meeting Chi-Square test assumptions, such as sufficiently large expected frequencies.

## Recommendations

Though state-based differences in churn are insignificant, stakeholders should explore other factors, such as customer satisfaction and service quality, that may influence churn. Potential actions include:
- Analyzing customer feedback for dissatisfaction patterns
- Investigating how pricing and service offerings correlate with churn
- Enhancing customer support to improve retention broadly

## Requirements

This project utilizes Python with the following libraries:
- **pandas**: data manipulation
- **matplotlib**: data visualization
- **scipy.stats**: statistical testing