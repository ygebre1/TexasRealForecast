from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import pandas as pd
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train_forecast_model(data):
    """Train the SARIMAX model on prepared data"""
    y = data['median_sale_price']
    X = data[['migration_lag1', 'migration_rolling_12m', 'migration_rolling_6m',
              'median_list_price_lag1', 'median_ppsf_lag1', 'inventory_lag1',
              'median_list_price_rolling_6m', 'CPI_lag1', 'Mortgage_rate_lag1',
              'Mortgage_rate_rolling_12m', 'CPI_rolling_12m']]
    
    # Auto ARIMA to find best parameters
    auto_model = pm.auto_arima(
        y,
        exog=X,
        seasonal=True,
        m=12,
        d=1,
        D=1,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        n_jobs=-1
    )
    
    # Train final model
    model = SARIMAX(
        y,
        exog=X,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    return model

def forecast_external_variable(data, variable, steps=48):
    """Forecast an external variable using its own SARIMAX model"""
    auto_model = pm.auto_arima(
        data[variable],
        seasonal=True,
        m=12,
        d=1,
        D=1,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        n_jobs=-1
    )
    
    model = SARIMAX(
        data[variable],
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False
    ).fit(disp=False)
    
    return model.get_forecast(steps=steps).predicted_mean

def get_forecasts(data, model, steps=48):
    """Get future forecasts for all external variables and then the target"""
    # Forecast each external variable
    forecasts = {}
    for var in ['median_list_price', 'median_ppsf', 'inventory', 'CPI', 'Mortgage_rate', 'migration']:
        forecasts[var] = forecast_external_variable(data, var, steps)
    
    # Generate date range for forecasted periods
    forecast_dates = pd.date_range(start="2024-12-01", periods=steps, freq='M')

    # Combine forecasts into a DataFrame
    forecast_df = pd.DataFrame({
        'period_begin': forecast_dates,
        **forecasts
    }).set_index('period_begin')

    # 🧠 Pad with the last 12 rows of historical data to calculate lag/rolling features
    padding = data[-12:].copy()
    combined = pd.concat([padding, forecast_df])

    # Lag and rolling features (applied to the combined DataFrame)
    combined['median_list_price_lag1'] = combined['median_list_price'].shift(1)
    combined['median_ppsf_lag1'] = combined['median_ppsf'].shift(1)
    combined['inventory_lag1'] = combined['inventory'].shift(1)
    combined['median_list_price_rolling_6m'] = combined['median_list_price'].shift(1).rolling(6).mean()
    combined['migration_lag1'] = combined['migration'].shift(1)
    combined['migration_rolling_12m'] = combined['migration'].shift(1).rolling(12).mean()
    combined['migration_rolling_6m'] = combined['migration'].shift(1).rolling(6).mean()
    combined['CPI_lag1'] = combined['CPI'].shift(1)
    combined['Mortgage_rate_lag1'] = combined['Mortgage_rate'].shift(1)
    combined['Mortgage_rate_rolling_12m'] = combined['Mortgage_rate'].shift(1).rolling(12).mean()
    combined['CPI_rolling_12m'] = combined['CPI'].shift(1).rolling(12).mean()

    # Slice forecast features only (exclude the padded historical part)
    exog_vars = ['migration_lag1', 'migration_rolling_12m', 'migration_rolling_6m',
                 'median_list_price_lag1', 'median_ppsf_lag1', 'inventory_lag1',
                 'median_list_price_rolling_6m', 'CPI_lag1', 'Mortgage_rate_lag1',
                 'Mortgage_rate_rolling_12m', 'CPI_rolling_12m']

    forecast_exog = combined.loc[forecast_df.index, exog_vars]

    # Final forecast
    forecast = model.get_forecast(steps=steps, exog=forecast_exog)

    # Combine historical and forecasted data
    full_data = pd.concat([data, combined.loc[forecast_df.index]])

    return full_data, forecast.predicted_mean
