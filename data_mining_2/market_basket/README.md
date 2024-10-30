# Market Basket Analysis for Cross-Selling Strategy

This project employs market basket analysis to identify product combinations frequently purchased together. By analyzing transaction data, this approach enables the development of targeted cross-selling strategies, creating personalized product recommendations, and optimizing inventory to improve customer satisfaction and increase revenue.

## Project Overview

### Objective
The primary question this analysis addresses is:
**"Which accessories are commonly purchased together?"**

### Goals
1. **Identify Product Bundles**: Reveal common product combinations for strategic bundling.
2. **Improve Customer Experience**: Use data insights to personalize recommendations and enhance shopping convenience.
3. **Boost Sales**: Drive cross-selling through targeted promotions based on frequently purchased items.

## Methodology

### Market Basket Analysis
The analysis uses association rule mining with the Apriori algorithm to identify relationships among items in transactions. The key metrics evaluated are:
- **Support**: Frequency of item combinations.
- **Confidence**: Likelihood of buying one item given the purchase of another.
- **Lift**: Strength of the association compared to random chance.

### Top Rules
1. **High Confidence Rule**: Indicates a strong likelihood of certain items being purchased together.
2. **High Lift Rule**: Demonstrates a significant association, ideal for cross-selling.
3. **High Support Rule**: Highlights popular itemsets for potential bundling.

### Libraries Used
- **pandas**: Data manipulation.
- **mlxtend**: Apriori algorithm for association rule mining.
- **matplotlib**: Visualization.

## Key Results and Insights

### Top Item Associations
- **High Demand Pairings**: Items like "10ft iPhone Charger Cable 2 Pack" and "Dust-Off Compressed Gas 2 pack" have high confidence and lift, indicating strong cross-selling opportunities.
- **Popular Bundles**: Frequently purchased combinations, such as "VIVO Dual LCD Monitor Desk mount" with "SanDisk Ultra 64GB card," are ideal for promotions and bundling.

### Practical Applications
1. **Personalized Recommendations**: Suggest related items in the shopping cart, improving customer experience.
2. **Targeted Marketing Campaigns**: Run promotions on common pairings to encourage larger purchases.
3. **Inventory Optimization**: Maintain adequate stock levels of commonly paired items to prevent missed sales opportunities.

## Course of Action

1. **Create Cross-Selling Bundles**: Offer discounts for item combinations with high lift and confidence.
2. **Implement Personalized Recommendations**: Use confidence rules to recommend items based on customer behavior.
3. **Optimize Stock**: Ensure adequate inventory for high-support items frequently purchased together.

## Files and Dependencies

- **data_cleaning.ipynb**: Jupyter notebook containing data transformation, market basket analysis, and association rule mining.
- **market_basket_analysis.pdf**: Documentation detailing methodology, analysis findings, and actionable recommendations.