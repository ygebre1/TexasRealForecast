import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

def prepare_mlp_data(selected_metro):
    # Load and merge data
    texas = pd.read_csv("data/texas_real_estate_data.csv")
    mig = pd.read_csv("data/migration-complete-data.csv")
    macro = pd.read_csv("data/macro_new_data.csv")
    
    texas['period_begin'] = texas['period_begin'].astype(str)
    mig['period_begin'] = mig['period_begin'].astype(str)
    macro['date'] = macro['date'].astype(str)
    
    texas_df0 = texas.merge(mig, on='period_begin', how='left')
    texas_df = texas_df0.merge(macro[['date', 'CPI', 'Mortgage_rate']], 
                              left_on='period_begin', right_on='date', how='left')
    
    # Filter for selected metro
    texas_df = texas_df[texas_df['parent_metro_region'] == selected_metro]
    
    # Feature engineering
    texas_df['sq_ft'] = texas_df['median_sale_price'] / texas_df['median_ppsf']
    texas_df['sq_ft'] = texas_df['sq_ft'].replace([np.inf, -np.inf], np.nan)
    texas_df['period_begin'] = pd.to_datetime(texas_df['period_begin'])
    texas_df['year'] = texas_df['period_begin'].dt.year
    
    # Split data
    df_train = texas_df[texas_df['year'] < 2023]
    df_test = texas_df[texas_df['year'] == 2023]
    df_2024 = texas_df[texas_df['year'] == 2024]
    
    # Encode categorical variables
    city_mean_price_train = df_train.groupby('city')['median_sale_price'].mean()
    df_train['city_en'] = df_train['city'].map(city_mean_price_train)
    df_train['city_en'] = df_train['city_en'].fillna(df_train['city_en'].mean())
    
    parent_metro_region_mean_price_train = df_train.groupby('parent_metro_region')['median_sale_price'].mean()
    df_train['parent_metro_region_en'] = df_train['parent_metro_region'].map(parent_metro_region_mean_price_train)
    df_train['parent_metro_region_en'] = df_train['parent_metro_region_en'].fillna(df_train['parent_metro_region_en'].mean())
    
    # One-Hot Encoding
    df_train = pd.get_dummies(df_train, columns=['property_type'], drop_first=True)
    
    # Prepare test data similarly
    df_test['city_en'] = df_test['city'].map(city_mean_price_train)  # Use train mapping
    df_test['city_en'] = df_test['city_en'].fillna(df_test['city_en'].mean())
    df_test['parent_metro_region_en'] = df_test['parent_metro_region'].map(parent_metro_region_mean_price_train)
    df_test['parent_metro_region_en'] = df_test['parent_metro_region_en'].fillna(df_test['parent_metro_region_en'].mean())
    df_test = pd.get_dummies(df_test, columns=['property_type'], drop_first=True)
    
    # Prepare 2024 data
    df_2024['city_en'] = df_2024['city'].map(city_mean_price_train)  # Use train mapping
    df_2024['city_en'] = df_2024['city_en'].fillna(df_2024['city_en'].mean())
    df_2024['parent_metro_region_en'] = df_2024['parent_metro_region'].map(parent_metro_region_mean_price_train)
    df_2024['parent_metro_region_en'] = df_2024['parent_metro_region_en'].fillna(df_2024['parent_metro_region_en'].mean())
    df_2024 = pd.get_dummies(df_2024, columns=['property_type'], drop_first=True)
    
    # Feature columns
    feature_cols = ['city_en', 'property_type_Multi-Family (2-4 Unit)', 
                   'property_type_Single Family Residential', 'property_type_Townhouse', 
                   'median_sale_price_mom', 'median_sale_price_yoy', 'median_list_price',
                   'median_ppsf', 'median_ppsf_yoy', 'median_list_ppsf', 'homes_sold', 
                   'homes_sold_yoy', 'pending_sales', 'pending_sales_yoy', 'new_listings', 
                   'new_listings_yoy', 'inventory', 'inventory_yoy', 'median_dom',
                   'median_dom_yoy', 'parent_metro_region_en', 'sq_ft', 'migration']
    
    # Prepare data
    X_train = df_train[feature_cols]
    y_train = df_train['median_sale_price']
    X_test = df_test[feature_cols]
    y_test = df_test['median_sale_price']
    X_2024 = df_2024[feature_cols]
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_2024_scaled = scaler.transform(X_2024)
    
    return df_train, df_test, df_2024, X_train_scaled, y_train, X_test_scaled, y_test, X_2024_scaled, scaler

def train_and_save_mlp_models():
    # Load metro regions
    texas = pd.read_csv("data/texas_real_estate_data.csv")
    metro_regions = texas['parent_metro_region'].unique()
    
    # Create directory for saved models if it doesn't exist
    os.makedirs("mlp_models", exist_ok=True)
    
    for metro in metro_regions:
        print(f"Training MLP model for {metro}...")
        
        try:
            # Prepare data
            df_train, df_test, df_2024, X_train_scaled, y_train, _, _, _, scaler = prepare_mlp_data(metro)
            
            # Train MLP model
            mlp_model = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            mlp_model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            metro_name = metro.split(',')[0].replace(' ', '_')
            model_filename = f"mlp_models/{metro_name}_TX_mlp_model.pkl"
            scaler_filename = f"mlp_models/{metro_name}_TX_scaler.pkl"
            
            with open(model_filename, 'wb') as f:
                pickle.dump(mlp_model, f)
                
            with open(scaler_filename, 'wb') as f:
                pickle.dump(scaler, f)
                
            print(f"Saved model for {metro} to {model_filename}")
            
        except Exception as e:
            print(f"Error processing {metro}: {str(e)}")
            continue

if __name__ == '__main__':
    train_and_save_mlp_models()
    print("MLP model training and saving complete!")