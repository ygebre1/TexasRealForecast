import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import os
import pickle

# Suppress warnings for cleaner output
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

# Initialize the Dash app with external stylesheets
app = dash.Dash(__name__, title="Texas Real Estate Forecast",external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server  # For deployment

# Custom CSS styles
CUSTOM_STYLES = {
    'header': {
        'backgroundColor': '#2c3e50',
        'color': 'white',
        'padding': '1.5rem',
        'marginBottom': '1.5rem',
        'borderRadius': '5px',
        'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)'
    },
    'dropdown': {
        'backgroundColor': '#f8f9fa',
        'borderRadius': '4px',
        'padding': '10px',
        'boxShadow': '0 2px 3px 0 rgba(0, 0, 0, 0.1)'
    },
    'card': {
        'backgroundColor': 'white',
        'borderRadius': '5px',
        'padding': '20px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 3px 0 rgba(0, 0, 0, 0.1)'
    },
    'tab': {
        'border': '1px solid #d6d6d6',
        'padding': '6px',
        'backgroundColor': '#f8f9fa'
    },
    'tab-selected': {
        'borderTop': '3px solid #2c3e50',
        'borderBottom': '1px solid #d6d6d6',
        'backgroundColor': 'white',
        'color': '#2c3e50',
        'padding': '6px',
        'fontWeight': 'bold'
    }
}

# Load your data
real_estate = pd.read_csv("data/texas_real_estate_data.csv")
macro = pd.read_csv('data/macro_new_data.csv')
mig = pd.read_csv("data/monthly-migration copy 1.csv")
mig['period_begin'] = pd.to_datetime(mig['period_begin'], format='%m/%d/%Y')

# Get unique metro regions from the real estate data
metro_regions = real_estate['parent_metro_region'].unique()

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Texas Real Estate Forecasting Dashboard", style={'marginBottom': '0'}),
        html.P("Interactive dashboard for forecasting real estate trends across Texas metro areas", 
              style={'marginTop': '5px', 'opacity': '0.8'})
    ], style=CUSTOM_STYLES['header']),
    
    html.Div([
        html.Div([
            html.Div([
                html.Label("Select Metro Region:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='metro-dropdown',
                    options=[{'label': region, 'value': region} for region in metro_regions],
                    value='Austin, TX' if 'Austin, TX' in metro_regions else metro_regions[0],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style=CUSTOM_STYLES['dropdown']),
        ], className="six columns"),
        
        html.Div([
            html.Div([
                html.Label("Forecast Horizon:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='forecast-horizon',
                    options=[
                        {'label': '12 months', 'value': 24},
                        {'label': '24 months', 'value': 36},
                        {'label': '36 months', 'value': 48},
                        {'label': '48 months', 'value': 60}
                    ],
                    value=24,
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style=CUSTOM_STYLES['dropdown']),
        ], className="six columns"),
    ], className="row", style={'marginBottom': '20px'}),
    
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label='Historical & Forecast', children=[
                    html.Div([
                        dcc.Graph(id='combined-graph', style={'height': '500px'})
                    ], style=CUSTOM_STYLES['card']),
                    html.Div(id='model-summary', style=CUSTOM_STYLES['card'])
                ], style=CUSTOM_STYLES['tab'], selected_style=CUSTOM_STYLES['tab-selected']),
                
                dcc.Tab(label='External Factors Forecast', children=[
                    html.Div([
                        dcc.Graph(id='external-forecasts-graph', style={'height': '500px'})
                    ], style=CUSTOM_STYLES['card']),
                    html.Div([
                        html.H4("About External Factors"),
                        html.P("""
                            This tab shows forecasted values for key external factors that influence real estate prices:
                            Migration patterns, mortgage rates, and CPI (Consumer Price Index). These forecasts are used 
                            as inputs for the main price prediction model.
                        """, style={'opacity': '0.8'})
                    ], style=CUSTOM_STYLES['card'])
                ], style=CUSTOM_STYLES['tab'], selected_style=CUSTOM_STYLES['tab-selected']),
                
                dcc.Tab(label='Model Diagnostics', children=[
                    html.Div(id='model-params', style=CUSTOM_STYLES['card']),
                    html.Div([
                        dcc.Graph(id='residuals-graph', style={'height': '400px'})
                    ], style=CUSTOM_STYLES['card']),
                    html.Div([
                        html.H4("About Model Diagnostics"),
                        html.P("""
                            The residuals plot helps assess model quality. Ideally, residuals should be randomly 
                            scattered around zero without visible patterns. The model parameters show the specific 
                            configuration of the SARIMAX model used for forecasting.
                        """, style={'opacity': '0.8'})
                    ], style=CUSTOM_STYLES['card'])
                ], style=CUSTOM_STYLES['tab'], selected_style=CUSTOM_STYLES['tab-selected'])
            ])
        ]
    ),
    
    html.Footer([
        html.P("Texas Real Estate Dashboard v1.0", style={'textAlign': 'center', 'marginTop': '20px'}),
    ], style={'marginTop': '30px'})
])

# Callback to update all components
@app.callback(
    [Output('combined-graph', 'figure'),
     Output('external-forecasts-graph', 'figure'),
     Output('model-params', 'children'),
     Output('residuals-graph', 'figure'),
     Output('model-summary', 'children')],
    [Input('metro-dropdown', 'value'),
     Input('forecast-horizon', 'value')]
)
def update_dashboard(selected_metro, forecast_horizon):
    # 1. Process data for selected metro region
    pr = real_estate[real_estate['parent_metro_region'] == selected_metro]
    
    monthly_selected = pr.groupby('period_begin').agg({
        'median_sale_price': 'mean',
        'median_list_price': 'mean',
        'homes_sold': 'sum',
        'inventory': 'sum',
        'pending_sales': 'sum',
        'new_listings': 'sum',
        'median_dom': 'mean',
        'median_ppsf': 'mean'
    }).reset_index()
    
    monthly_selected['period_begin'] = monthly_selected['period_begin'].astype(str)
    combined = monthly_selected.merge(macro, left_on='period_begin', right_on='date', how='left')
    combined['period_begin'] = combined['period_begin'].astype(str)
    mig['period_begin'] = mig['period_begin'].astype(str)
    final = combined.merge(mig, on='period_begin', how='left')
    
    # 2. Feature engineering
    final_macro_mig = final.copy()
    final_macro_mig['median_list_price_lag1'] = final_macro_mig['median_list_price'].shift(1)
    final_macro_mig['median_ppsf_lag1'] = final_macro_mig['median_ppsf'].shift(1)
    final_macro_mig['inventory_lag1'] = final_macro_mig['inventory'].shift(1)
    final_macro_mig['median_list_price_rolling_6m'] = final_macro_mig['median_list_price'].shift(1).rolling(6).mean()
    final_macro_mig['migration_lag1'] = final_macro_mig['migration'].shift(1)
    final_macro_mig['migration_rolling_12m'] = final_macro_mig['migration'].shift(1).rolling(12).mean()
    final_macro_mig['migration_rolling_6m'] = final_macro_mig['migration'].shift(1).rolling(6).mean()
    final_macro_mig['CPI_lag1'] = final_macro_mig['CPI'].shift(1)
    final_macro_mig['Mortgage_rate_lag1'] = final_macro_mig['Mortgage_rate'].shift(1)
    final_macro_mig['Mortgage_rate_rolling_12m'] = final_macro_mig['Mortgage_rate'].shift(1).rolling(12).mean()
    final_macro_mig['CPI_rolling_12m'] = final_macro_mig['CPI'].shift(1).rolling(12).mean()
    
    final_macro_mig['period_begin'] = pd.to_datetime(final_macro_mig['period_begin'])
    final_macro_mig = final_macro_mig.set_index('period_begin')
    final_macro_mig_no_null = final_macro_mig.dropna()
    
    # 3. Train the main SARIMAX model
    auto_model_macro_mig = pm.auto_arima(
        final_macro_mig_no_null['median_sale_price'],
        exog=final_macro_mig_no_null[['migration_lag1', 'migration_rolling_12m', 'migration_rolling_6m',
                                    'median_list_price_lag1', 'median_ppsf_lag1', 'inventory_lag1',
                                    'median_list_price_rolling_6m', 'CPI_lag1', 'Mortgage_rate_lag1',
                                    'Mortgage_rate_rolling_12m', 'CPI_rolling_12m']],
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
    
    # Get model parameters
    order = auto_model_macro_mig.order
    seasonal_order = auto_model_macro_mig.seasonal_order
    
    # Fit the final model
    sarimax_model_macro_mig_full = SARIMAX(
        final_macro_mig_no_null['median_sale_price'],
        exog=final_macro_mig_no_null[['migration_lag1', 'migration_rolling_12m', 'migration_rolling_6m',
                                     'median_list_price_lag1', 'median_ppsf_lag1', 'inventory_lag1',
                                     'median_list_price_rolling_6m', 'CPI_lag1', 'Mortgage_rate_lag1',
                                     'Mortgage_rate_rolling_12m', 'CPI_rolling_12m']],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # 4. Load precomputed external forecasts from file
    external_vars = ['median_list_price', 'median_ppsf', 'inventory', 'CPI', 'Mortgage_rate', 'migration']
    city_name = selected_metro.split(',')[0].strip().replace(' ', '_')  # Handle spaces in names
    forecast_file_path = os.path.join("external_forecasts", f"{city_name}_TX.pkl")

    # Load forecasts from file
    if not os.path.exists(forecast_file_path):
        raise FileNotFoundError(f"Forecast file not found for {selected_metro}: {forecast_file_path}")

    with open(forecast_file_path, "rb") as f:
        external_forecasts = pickle.load(f)

    # Ensure all required variables are present
    missing_vars = [var for var in external_vars if var not in external_forecasts]
    if missing_vars:
        raise KeyError(f"Missing variables in external forecast: {missing_vars}")

    # 5. Prepare future dates and build future exogenous DataFrame
    future_dates = pd.date_range(
        start=final_macro_mig_no_null.index[-1] + pd.DateOffset(months=1),
        periods=forecast_horizon,
        freq='MS'
    )

    future_exog = pd.DataFrame(index=future_dates)
    for var in external_vars:
        future_exog[var] = external_forecasts[var].values[:forecast_horizon]

    # Calculate necessary lags and rolling features
    future_exog['median_list_price_lag1'] = future_exog['median_list_price'].shift(1)
    future_exog['median_ppsf_lag1'] = future_exog['median_ppsf'].shift(1)
    future_exog['inventory_lag1'] = future_exog['inventory'].shift(1)
    future_exog['median_list_price_rolling_6m'] = future_exog['median_list_price'].shift(1).rolling(6).mean()
    future_exog['migration_lag1'] = future_exog['migration'].shift(1)
    future_exog['migration_rolling_12m'] = future_exog['migration'].shift(1).rolling(12).mean()
    future_exog['migration_rolling_6m'] = future_exog['migration'].shift(1).rolling(6).mean()
    future_exog['CPI_lag1'] = future_exog['CPI'].shift(1)
    future_exog['Mortgage_rate_lag1'] = future_exog['Mortgage_rate'].shift(1)
    future_exog['Mortgage_rate_rolling_12m'] = future_exog['Mortgage_rate'].shift(1).rolling(12).mean()
    future_exog['CPI_rolling_12m'] = future_exog['CPI'].shift(1).rolling(12).mean()

    # Drop NA from lag/rolling
    future_exog = future_exog.dropna()

    # 6. Generate the main forecast
    forecast = sarimax_model_macro_mig_full.get_forecast(
        steps=len(future_exog),
        exog=future_exog[['migration_lag1', 'migration_rolling_12m', 'migration_rolling_6m',
                         'median_list_price_lag1', 'median_ppsf_lag1', 'inventory_lag1',
                         'median_list_price_rolling_6m', 'CPI_lag1', 'Mortgage_rate_lag1',
                         'Mortgage_rate_rolling_12m', 'CPI_rolling_12m']]
    )
    
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # 7. Create figures with enhanced styling
    combined_fig = go.Figure()
    
    # Historical data
    combined_fig.add_trace(go.Scatter(
        x=final_macro_mig_no_null.index,
        y=final_macro_mig_no_null['median_sale_price'],
        name='Historical Median Sale Price',
        line=dict(color='#3498db', width=2.5),
        mode='lines',
        hovertemplate='%{x|%b %Y}<br>Price: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add connecting line between last historical point and first forecast point
    combined_fig.add_trace(go.Scatter(
        x=[final_macro_mig_no_null.index[-1], forecast_mean.index[0]],
        y=[final_macro_mig_no_null['median_sale_price'].iloc[-1], forecast_mean.iloc[0]],
        line=dict(color='#3498db', dash='dot', width=1.5),
        mode='lines',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Forecast
    combined_fig.add_trace(go.Scatter(
        x=forecast_mean.index,
        y=forecast_mean,
        name='Forecast',
        line=dict(color='#2ecc71', dash='dash', width=2.5),
        mode='lines',
        hovertemplate='%{x|%b %Y}<br>Forecast: $%{y:,.0f}<extra></extra>'
    ))
    
    # Confidence interval
    combined_fig.add_trace(go.Scatter(
        x=forecast_ci.index.tolist() + forecast_ci.index[::-1].tolist(),
        y=forecast_ci.iloc[:, 0].tolist() + forecast_ci.iloc[:, 1][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    combined_fig.update_layout(
        title=f'Median Sale Price Forecast - {selected_metro}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(245, 246, 249, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # External factors forecast figure
    external_fig = go.Figure()
    
    colors = ['#e74c3c', '#9b59b6', '#f39c12']
    for i, var in enumerate(['migration', 'Mortgage_rate', 'CPI']):
        external_fig.add_trace(go.Scatter(
            x=future_dates[:len(external_forecasts[var])],
            y=external_forecasts[var][:forecast_horizon],
            name=var.replace('_', ' ').title(),
            line=dict(color=colors[i], width=2),
            mode='lines',
            hovertemplate='%{x|%b %Y}<br>Value: %{y:.2f}<extra></extra>'
        ))
    
    external_fig.update_layout(
        title=f'External Factors Forecast - {selected_metro}',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(245, 246, 249, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Residuals plot
    residuals = sarimax_model_macro_mig_full.resid
    residuals_fig = go.Figure()
    residuals_fig.add_trace(go.Scatter(
        x=final_macro_mig_no_null.index,
        y=residuals,
        name='Residuals',
        mode='markers',
        marker=dict(color='#3498db', size=8, opacity=0.6)
    ))
    residuals_fig.add_hline(y=0, line_dash="dash", line_color="red")
    residuals_fig.update_layout(
        title='Model Residuals',
        xaxis_title='Date',
        yaxis_title='Residuals',
        plot_bgcolor='rgba(245, 246, 249, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    # Model parameters text with better formatting
    last_date = final_macro_mig_no_null.index[-1].strftime('%Y-%m-%d')
    model_params = [
        html.H3("Model Parameters", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
        html.Div([
            html.Div([
                html.P(html.B("Selected Metro:"), style={'marginBottom': '5px'}),
                html.P(selected_metro, style={'marginLeft': '15px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("SARIMAX Order:"), style={'marginBottom': '5px'}),
                html.P(str(order), style={'marginLeft': '15px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Seasonal Order:"), style={'marginBottom': '5px'}),
                html.P(str(seasonal_order), style={'marginLeft': '15px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Model Fit Statistics:"), style={'marginBottom': '5px'}),
                html.P(f"AIC: {sarimax_model_macro_mig_full.aic:.2f}", style={'marginLeft': '15px'}),
                html.P(f"BIC: {sarimax_model_macro_mig_full.bic:.2f}", style={'marginLeft': '15px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Last Training Date:"), style={'marginBottom': '5px'}),
                html.P(last_date, style={'marginLeft': '15px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Forecast Horizon:"), style={'marginBottom': '5px'}),
                html.P(f"{forecast_horizon} months", style={'marginLeft': '15px'})
            ])
        ])
    ]
    
    # Model summary statistics with better formatting
    last_price = final_macro_mig_no_null['median_sale_price'].iloc[-1]
    forecast_start_price = forecast_mean.iloc[0]
    forecast_end_price = forecast_mean.iloc[-1]
    pct_change = ((forecast_end_price - last_price) / last_price) * 100
    
    # Determine if change is positive or negative for styling
    change_color = '#2ecc71' if pct_change >= 0 else '#e74c3c'
    change_icon = '↑' if pct_change >= 0 else '↓'
    
    model_summary = [
        html.H3("Forecast Summary", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
        html.Div([
            html.Div([
                html.P(html.B("Last Historical Price:"), style={'marginBottom': '5px'}),
                html.P(f"${last_price:,.0f}", style={'marginLeft': '15px', 'fontSize': '18px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Starting Forecast Price:"), style={'marginBottom': '5px'}),
                html.P(f"${forecast_start_price:,.0f}", style={'marginLeft': '15px', 'fontSize': '18px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Ending Forecast Price:"), style={'marginBottom': '5px'}),
                html.P(f"${forecast_end_price:,.0f}", style={'marginLeft': '15px', 'fontSize': '18px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Total Projected Change:"), style={'marginBottom': '5px'}),
                html.P(f"{change_icon} {abs(pct_change):.1f}%", 
                      style={'marginLeft': '15px', 'color': change_color, 'fontSize': '18px'})
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.P(html.B("Average Monthly Change:"), style={'marginBottom': '5px'}),
                html.P(f"{change_icon} {abs(pct_change/forecast_horizon):.2f}%", 
                      style={'marginLeft': '15px', 'color': change_color})
            ])
        ])
    ]

    return (combined_fig, external_fig, model_params, residuals_fig, model_summary)
    

# Run the dash app
if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8080)