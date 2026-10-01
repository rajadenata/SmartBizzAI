import pandas as pd
import numpy as np
import os
import gc

class M5DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.sales_path = os.path.join(data_path, 'sales_train_validation.csv')
        self.calendar_path = os.path.join(data_path, 'calendar.csv')
        self.prices_path = os.path.join(data_path, 'sell_prices.csv')

    def reduce_mem_usage(self, df):
        """Iterate through all columns of a dataframe and modify the data type to reduce memory usage."""
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype

            # Skip already-optimised or non-numeric types
            if str(col_type) == 'category':
                continue
            if pd.api.types.is_datetime64_any_dtype(col_type):
                continue
            if pd.api.types.is_string_dtype(col_type) or col_type == object:
                # Convert plain object/string columns to category (except raw date strings)
                if col not in ['date', 'd']:
                    df[col] = df[col].astype('category')
                continue

            # Numeric downcast
            try:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            except (TypeError, ValueError):
                # Cannot determine numeric bounds — leave column as-is
                pass

        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f'Memory usage decreased to {end_mem:.2f} Mb ({reduction:.1f}% reduction)')
        return df


    def validate_paths(self):
        for p in [self.sales_path, self.calendar_path, self.prices_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required M5 file: {p}")
        print("All dataset files found.")

    def load_and_aggregate(self):
        self.validate_paths()
        
        metadata = {}
        
        print("Loading calendar...")
        calendar = pd.read_csv(self.calendar_path)
        calendar = self.reduce_mem_usage(calendar)
        
        print("Loading prices...")
        prices = pd.read_csv(self.prices_path)
        prices = self.reduce_mem_usage(prices)
        
        print("Loading sales (This may take a moment)...")
        sales = pd.read_csv(self.sales_path)
        
        metadata['original_item_series'] = len(sales)
        metadata['n_stores'] = sales['store_id'].nunique()
        metadata['n_categories'] = sales['cat_id'].nunique()
        
        print("Aggregating sales to Store & Category level...")
        d_cols = [c for c in sales.columns if c.startswith('d_')]
        agg_sales = sales.groupby(['store_id', 'cat_id'])[d_cols].sum().reset_index()
        
        print("Melting sales data...")
        agg_sales = agg_sales.melt(id_vars=['store_id', 'cat_id'], var_name='d', value_name='demand')
        agg_sales = self.reduce_mem_usage(agg_sales)
        
        print("Aggregating prices to Store & Category level...")
        item_cats = sales[['item_id', 'cat_id']].drop_duplicates()
        del sales
        gc.collect()

        avg_prices = prices.merge(item_cats, on='item_id', how='left')
        avg_prices = avg_prices.groupby(['store_id', 'cat_id', 'wm_yr_wk'])['sell_price'].mean().reset_index()
        avg_prices = self.reduce_mem_usage(avg_prices)
        del prices, item_cats
        gc.collect()

        print("Merging calendar and price features...")
        data = agg_sales.merge(calendar[['d', 'date', 'wm_yr_wk', 'weekday', 'month', 'year']], on='d', how='left')
        data = data.merge(avg_prices, on=['store_id', 'cat_id', 'wm_yr_wk'], how='left')
        
        print("Handling missing values...")
        # Fill NA for prices (forward fill only to prevent future leakage)
        data['sell_price'] = data.groupby(['store_id', 'cat_id'])['sell_price'].ffill()
        # Do not back-fill or fabricate historical prices.
        # Any NaNs (unseen prices) will be dropped during feature engineering to ensure no leakage.
        
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values(['store_id', 'cat_id', 'date'], inplace=True)
        data = data.reset_index(drop=True)
        
        data = self.reduce_mem_usage(data)
        print(f"Final aggregated dataset shape: {data.shape}")
        
        metadata['actual_aggregated_rows'] = len(data)
        metadata['n_dates'] = data['date'].nunique()
        metadata['actual_store_category_series'] = data[['store_id', 'cat_id']].drop_duplicates().shape[0]
        metadata['series_reduction_factor'] = metadata['original_item_series'] / metadata['actual_store_category_series'] if metadata['actual_store_category_series'] > 0 else 0
        
        # Duplicate check and missing-date check
        metadata['duplicate_records'] = data.duplicated(subset=['store_id', 'cat_id', 'date']).sum()
        
        print("Validating missing dates per series...")
        # Check each Store x Category series individually for continuous date sequence
        global_dates = set(data['date'])
        
        series_missing_dates_count = 0
        total_missing_dates = 0
        grouped_dates = data.groupby(['store_id', 'cat_id'])['date']
        
        for name, group in grouped_dates:
            actual_dates = set(group)
            if actual_dates != global_dates:
                series_missing_dates_count += 1
                total_missing_dates += len(global_dates) - len(actual_dates)

        metadata['number_of_series_checked'] = metadata['actual_store_category_series']
        metadata['series_with_missing_dates'] = series_missing_dates_count
        metadata['missing_dates'] = total_missing_dates
        
        return data, metadata
