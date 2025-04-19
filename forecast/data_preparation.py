import pandas as pd
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_data(real_estate_path, macro_path, mig_path, metro_region='Austin, TX'):
    """Prepare the combined dataset with all features"""
    # Load data
    real_estate = pd.read_csv(real_estate_path)
    macro = pd.read_csv(macro_path)
    mig = pd.read_csv(mig_path)

    # Convert 'period_begin' and 'date' columns to datetime format
    real_estate['period_begin'] = pd.to_datetime(real_estate['period_begin'], errors='coerce')
    mig['period_begin'] = pd.to_datetime(mig['period_begin'], format='%m/%d/%Y', errors='coerce')  # Migration date format

    # Ensure macro['date'] is also in datetime format (important for merging)
    macro['date'] = pd.to_datetime(macro['date'], errors='coerce')

    # Print the first few rows of each dataset to check if conversion worked
    print("Real Estate Data:")
    print(real_estate.head())
    print("Migration Data:")
    print(mig.head())
    print("Macro Data:")
    print(macro.head())

    # Process real estate data
    pr = real_estate[real_estate['parent_metro_region'] == metro_region]
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

    # Merge datasets (Ensure both 'period_begin' and 'date' are datetime)
    combined = monthly_selected.merge(macro, left_on='period_begin', right_on='date', how='left')
    print("After macro merge:")
    print(combined.head())

    final = combined.merge(mig, on='period_begin', how='left')
    print("After migration merge:")
    print(final.head())

    # Create lag and rolling features
    final['period_begin'] = pd.to_datetime(final['period_begin'])
    final = final.set_index('period_begin')
    final.index.freq = 'MS'

    # Lag features
    for col in ['median_list_price', 'median_ppsf', 'inventory', 'migration', 'CPI', 'Mortgage_rate']:
        final[f'{col}_lag1'] = final[col].shift(1)

    # Rolling features
    final['median_list_price_rolling_6m'] = final['median_list_price'].shift(1).rolling(6).mean()
    final['migration_rolling_12m'] = final['migration'].shift(1).rolling(12).mean()
    final['migration_rolling_6m'] = final['migration'].shift(1).rolling(6).mean()
    final['Mortgage_rate_rolling_12m'] = final['Mortgage_rate'].shift(1).rolling(12).mean()
    final['CPI_rolling_12m'] = final['CPI'].shift(1).rolling(12).mean()

    # Check if the final DataFrame is empty or not
    if final.empty:
        print("The final DataFrame is empty!")
    else:
        print("The final DataFrame has data:")
        print(final.head())

    return final.dropna()