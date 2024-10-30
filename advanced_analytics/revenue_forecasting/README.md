# Daily Revenue Forecasting Using ARIMA Model

This project employs time series analysis with the ARIMA model to forecast daily revenue for a telecommunications organization. The predictive model leverages historical data to support financial forecasting, enabling data-driven decisions for budget planning, resource allocation, and sales strategy optimization.

## Project Overview

### Research Question
**"Can we predict future daily revenue based on historical data?"**

### Goals
1. **Develop an Accurate Forecast**: Build a model that predicts daily revenue trends, assisting with strategic planning.
2. **Identify Revenue Patterns**: Analyze trends, seasonality, and autocorrelation in historical revenue data.
3. **Support Data-Driven Decision-Making**: Use forecasts to guide actions in workforce planning, budgeting, and marketing.

## Methodology

### Time Series Analysis
1. **Data Preparation**: Loaded, cleaned, and formatted daily revenue data over 731 days (January 1, 2021, to January 1, 2023).
2. **Stationarity Check**: Conducted the Augmented Dickey-Fuller test to assess stationarity; applied differencing to make the data stationary.
3. **ARIMA Model Selection**: Used ACF and PACF plots and `auto_arima` for optimal model parameters.
4. **Modeling**: Fitted a SARIMA(1,1,1)(1,1,1,12) model to capture both non-seasonal and seasonal components.
5. **Evaluation**: Split data into training and test sets; evaluated model accuracy with RMSE, achieving an RMSE of 2.177 million.

### Key Metrics
- **RMSE**: Measures average deviation of forecasted values from actual values.
- **Confidence Intervals**: Provides a range within which future revenue is expected to fall, reflecting uncertainty over time.

### Libraries
- **pandas**: Data manipulation.
- **pmdarima**: Automated ARIMA model selection.
- **matplotlib**: Visualization.
- **scikit-learn**: Train-test split and RMSE calculation.

## Key Findings

### Forecasting Results
- **Trend and Seasonality**: Revenue shows an upward trend with periodic seasonal patterns.
- **Prediction Accuracy**: RMSE of 2.177 million indicates forecast reliability for short-term planning.
- **Confidence Intervals**: Widen over time, underscoring increased uncertainty in long-term forecasts.

### Visualization Summary
1. **Time Series Plot**: Shows upward trend and seasonal fluctuations.
2. **Autocorrelation and Decomposition Plots**: Highlight seasonality and autocorrelation.
3. **Forecast Plot**: Displays forecasted values with confidence intervals for future revenue projections.

## Recommendations

1. **Strategic Decision-Making**: Use forecasts for budgeting, headcount adjustments, and promotional timing.
2. **Regular Model Updates**: Retrain the model monthly with new data for enhanced accuracy.
3. **Inventory and Resource Allocation**: Prepare contingency plans based on forecasted revenue ranges.

## Files and Dependencies

- **time_series_cleaning.ipynb**: Jupyter notebook containing data preparation, modeling, and forecasting code.
- **time_series.pdf**: Documentation detailing methodology, model selection, and analysis findings.