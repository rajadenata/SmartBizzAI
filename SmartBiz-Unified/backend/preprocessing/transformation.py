"""
Data transformation module for SmartBiz AI V1.0.
Handles normalization, scaling, aggregation, and train/test splitting
tailored to daily demand forecasting for culinary MSMEs.
"""

import pandas as pd
import numpy as np
import logging


def normalize_column(df, column, method="minmax"):
    """
    Normalize a numeric column using Min-Max or Z-score scaling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to normalize.
        method (str): "minmax" for [0,1] scaling, "zscore" for standardization.

    Returns:
        pd.DataFrame: DataFrame with the normalized column.
    """
    if df is None or df.empty or column not in df.columns:
        logging.warning(f"Cannot normalize: column '{column}' missing or DataFrame empty.")
        return df

    df = df.copy()

    if method == "minmax":
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max - col_min == 0:
            df[column] = 0.0
        else:
            df[column] = (df[column] - col_min) / (col_max - col_min)
        logging.info(f"Column '{column}' normalized (min-max).")

    elif method == "zscore":
        col_mean = df[column].mean()
        col_std = df[column].std()
        if col_std == 0:
            df[column] = 0.0
        else:
            df[column] = (df[column] - col_mean) / col_std
        logging.info(f"Column '{column}' standardized (z-score).")

    else:
        logging.warning(f"Unknown normalization method '{method}'.")

    return df


def aggregate_daily_sales(df, date_column="date", product_column="product_id",
                          target_column="sales_qty"):
    """
    Aggregate sales data to daily totals per product.

    Sums the target column and takes the max of the promotion flag per day.

    Args:
        df (pd.DataFrame): Sales DataFrame with date and product columns.
        date_column (str): Name of the date column.
        product_column (str): Name of the product identifier column.
        target_column (str): Name of the sales quantity column.

    Returns:
        pd.DataFrame: Aggregated daily sales DataFrame.
    """
    if df is None or df.empty:
        logging.warning("aggregate_daily_sales received an empty DataFrame.")
        return df

    df = df.copy()

    agg_dict = {target_column: "sum"}

    # Keep promotion flag (max = 1 if any promotion was active that day)
    if "promotion" in df.columns:
        agg_dict["promotion"] = "max"

    # Keep price as mean for the day
    if "price" in df.columns:
        agg_dict["price"] = "mean"

    df_agg = (
        df.groupby([date_column, product_column])
        .agg(agg_dict)
        .reset_index()
    )

    logging.info(
        f"Aggregated to {len(df_agg)} daily records "
        f"({df_agg[product_column].nunique()} products)."
    )
    return df_agg


def split_train_test(df, date_column="date", test_ratio=0.2):
    """
    Split time-series data into training and test sets using a temporal split.

    Unlike random splitting, this preserves chronological order so that the
    model is always trained on past data and evaluated on future data.

    Args:
        df (pd.DataFrame): Input DataFrame sorted by date.
        date_column (str): Name of the date column.
        test_ratio (float): Fraction of data to use as the test set (0.0 – 1.0).

    Returns:
        tuple: (train_df, test_df)
    """
    if df is None or df.empty:
        logging.warning("split_train_test received an empty DataFrame.")
        return df, pd.DataFrame()

    df = df.sort_values(by=date_column).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    logging.info(
        f"Train/test split: {len(train_df)} train rows, "
        f"{len(test_df)} test rows (ratio={test_ratio})."
    )
    return train_df, test_df


def filter_product(df, product_id, product_column="product_id"):
    """
    Filter a DataFrame to keep only rows for a specific product.

    Args:
        df (pd.DataFrame): Input DataFrame.
        product_id: The product identifier to keep.
        product_column (str): Name of the product column.

    Returns:
        pd.DataFrame: Filtered DataFrame for one product.
    """
    if df is None or df.empty:
        return df

    filtered = df[df[product_column] == product_id].reset_index(drop=True)
    logging.info(f"Filtered to product {product_id}: {len(filtered)} rows.")
    return filtered
