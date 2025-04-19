import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import pickle

# Load your datasets
macro = pd.read_csv('data/macro_new_data.csv')
mig = pd.read_csv("data/monthly-migration copy 1.csv")
real_estate = pd.read_csv("data/texas_real_estate_data.csv")

# Convert date columns to datetime
mig['period_begin'] = pd.to_datetime(mig['period_begin'], format='%m/%d/%Y')
macro['date'] = pd.to_datetime(macro['date'])

# External variables to forecast
external_vars = ['median_list_price', 'median_ppsf', 'inventory', 'CPI', 'Mortgage_rate', 'migration']
forecast_horizon = 60

# Where to save the forecasts
output_dir = "external_forecasts"
os.makedirs(output_dir, exist_ok=True)

# Loop through each metro and create forecasts
for metro in real_estate['parent_metro_region'].unique():
    print(f"\n📈 Processing {metro}...")

    # Prepare time series for this metro
    pr = real_estate[real_estate['parent_metro_region'] == metro]
    monthly = pr.groupby('period_begin').agg({
        'median_sale_price': 'mean',
        'median_list_price': 'mean',
        'homes_sold': 'sum',
        'inventory': 'sum',
        'pending_sales': 'sum',
        'new_listings': 'sum',
        'median_dom': 'mean',
        'median_ppsf': 'mean'
    }).reset_index()

    monthly['period_begin'] = pd.to_datetime(monthly['period_begin'])

    # Merge all external datasets
    combined = monthly.merge(macro, left_on='period_begin', right_on='date', how='left')
    combined = combined.merge(mig, on='period_begin', how='left')
    combined = combined.set_index('period_begin').dropna()

    # Forecast and store each external variable
    forecasts = {}
    for var in external_vars:
        try:
            print(f"   🔮 Forecasting {var}...")
            series = combined[var]

            # Fit SARIMA model
            model = pm.auto_arima(
                series,
                seasonal=True,
                m=12,
                d=1,
                D=1,
                start_p=0,
                start_q=0,
                max_p=3,
                max_q=3,
                start_P=0,
                start_Q=0,
                max_P=2,
                max_Q=2,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                n_jobs=-1
            )

            fitted_model = SARIMAX(
                series,
                order=model.order,
                seasonal_order=model.seasonal_order,
                enforce_stationarity=False
            ).fit(disp=False)

            pred = fitted_model.get_forecast(steps=forecast_horizon).predicted_mean
            forecasts[var] = pred

        except Exception as e:
            print(f"   ❌ Failed to forecast {var} for {metro}: {e}")
            continue

    # Save all forecasts to a .pkl file
    filename = f"{output_dir}/{metro.replace(',', '').replace(' ', '_')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(forecasts, f)

    print(f"✅ Saved forecast for {metro} to {filename}")
