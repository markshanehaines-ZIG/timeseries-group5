import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads the raw dataset."""
    print(f"Loading data from {filepath}...")
    df_raw = pd.read_csv(filepath)
    print(f"Raw dataset shape: {df_raw.shape}")
    return df_raw

def clean_data(df):
    """Performs the 4 data cleaning steps outlined in Exercise 1."""
    
    # 1. Convert Datetime column to datetime object
    print("Step 1: Converting 'Datetime' column to datetime objects...")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 2. Set datetime as index and sort chronologically
    print("Step 2: Setting index to 'Datetime' and sorting chronologically...")
    df = df.set_index('Datetime')
    df = df.sort_index()
    
    # 3. Identify and handle duplicate timestamps
    print("Step 3: Handling duplicate timestamps...")
    duplicates = df.index.duplicated(keep=False)
    n_dup_rows = duplicates.sum()
    print(f"  > Rows involved in duplicate timestamps: {n_dup_rows}")
    if n_dup_rows > 0:
        # Resolve by averaging values at the same timestamp
        df = df.groupby(df.index).mean()
        print(f"  > Shape after deduplication: {df.shape}")
        
    # 4. Force hourly frequency and fill gaps
    print("Step 4: Forcing hourly frequency and filling gaps via linear interpolation...")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    print(f"  > Expected hourly records: {len(full_range)}")
    print(f"  > Actual records:          {len(df)}")
    print(f"  > Missing hours (gaps):    {len(full_range) - len(df)}")
    
    # Reindex to force exact hourly frequency
    df = df.reindex(full_range)
    df.index.name = 'Datetime'
    
    # Fill gaps using linear interpolation (gradual change between known values)
    df['PJMW_MW'] = df['PJMW_MW'].interpolate(method='linear')
    
    print(f"  > Null values after interpolation: {df['PJMW_MW'].isnull().sum()}")
    print(f"Final dataset shape: {df.shape}")
    
    return df

def main():
    filepath = 'data/PJMW_hourly.csv'
    df_raw = load_data(filepath)
    df_cleaned = clean_data(df_raw.copy())
    
    # Optionally save the cleaned dataset
    df_cleaned.to_csv('data/PJMW_hourly_cleaned.csv')
    return df_cleaned

if __name__ == "__main__":
    main()
