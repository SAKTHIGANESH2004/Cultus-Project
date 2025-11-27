Advanced Time Series Forecasting with HAR-ARIMA

This project focuses on forecasting high-frequency time-series data using three models: HAR-ARIMA, SARIMA, and a feature-engineered XGBoost Regressor.
The aim is to evaluate how each model handles intraday, weekly, and seasonal behaviors.

Dataset:

An hourly dataset with clear intraday, weekly, and seasonal cycles.
Data was cleaned, time-indexed, and split into train/test sets.

Models Implemented:
HAR-ARIMA
Captures short-, medium-, and long-term dependencies through hierarchical lags.
SARIMA
Seasonal ARIMA with 24-hour seasonality modeling daily cycles.
XGBoost Regressor

Trained on engineered features such as lag values, rolling windows, time encodings, and Fourier terms.

               Final Model Performance:
Model         	MAE	             RMSE	         MAPE%
SARIMA      	9785.307688 	  11581.340783	  31.012689
HAR-ARIMA   	2024.443165   	2788.515336   	6.408976
XGBoost     	934.198378	    1249.355574	    3.115682



Visualizations Included:
Actual vs Predicted plots
Error distribution plots
Model comparison charts
Scatter plots for accuracy

Conclusion:

The results clearly show that XGBoost delivered the highest forecasting accuracy across all error metrics, thanks to flexible feature engineering and its ability to capture nonlinear relationships.
HAR-ARIMA performed significantly better than SARIMA, proving effective for multi-scale forecasting where short-term and long-term patterns coexist.
SARIMA, while stable for daily seasonality, struggled with broader temporal variations.
Overall, the project demonstrates that combining hierarchical modeling with modern machine-learning techniques yields stronger and more reliable time-series forecasts.
