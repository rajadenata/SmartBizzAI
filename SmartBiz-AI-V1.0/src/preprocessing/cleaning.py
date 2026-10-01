"""
Data cleaning module for SmartBiz AI V1.0.
Handles missing values, date parsing, outlier detection,
and duplicate removal for sales time-series data.
"""

import pandas as pd
import numpy as np
import logging


def handle_missing_values(df, strategy="interpolate"):
    """
    Handle missing values in the sales DataFrame.

    Strategies:
        - "interpolate": Linear interpolation for numeric columns,
          forward-fill for categorical columns.
        - "drop": Remove rows that contain any missing value.
        - "fill_zero": Replace all NaN with 0.
        - "ffill": Forward-fill all columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Strategy name (default: "interpolate").

    Returns:
        pd.DataFrame: Cleaned DataFrame with no missing values.
    """
    if df is None or df.empty:
        logging.warning("handle_missing_values received an empty DataFrame.")
        return df

    df = df.copy()
    missing_before = df.isnull().sum().sum()

    if strategy == "interpolate":
        # Numeric columns: linear interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

        # Non-numeric columns: forward-fill then backward-fill
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill()

    elif strategy == "drop":
        df = df.dropna()

    elif strategy == "fill_zero":
        df = df.fillna(0)

    elif strategy == "ffill":
        df = df.ffill().bfill()

    else:
        logging.warning(f"Unknown strategy '{strategy}', defaulting to 'interpolate'.")
        return handle_missing_values(df, strategy="interpolate")

    missing_after = df.isnull().sum().sum()
    logging.info(
        f"Missing values handled (strategy={strategy}): "
        f"{missing_before} → {missing_after}"
    )
    return df


def convert_date_column(df, date_column="date", date_format="%Y-%m-%d"):
    """
    Convert a date column to pandas datetime and sort by date.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_column (str): Name of the date column.
        date_format (str): Expected date format string.

    Returns:
        pd.DataFrame: DataFrame with parsed datetime column, sorted by date.
    """
    if df is None or df.empty:
        logging.warning("convert_date_column received an empty DataFrame.")
        return df

    df = df.copy()

    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        df = df.sort_values(by=date_column).reset_index(drop=True)
        logging.info(f"Date column '{date_column}' converted and sorted successfully.")
    except Exception as e:
        logging.error(f"Error converting date column '{date_column}': {e}")

    return df


def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list or None): Columns to consider for identifying duplicates.
            If None, all columns are used.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    rows_before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    rows_removed = rows_before - len(df)

    if rows_removed > 0:
        logging.info(f"Removed {rows_removed} duplicate row(s).")
    else:
        logging.info("No duplicates found.")

    return df


def detect_outliers(df, column, method="iqr", threshold=1.5):
    """
    Detect outliers in a numeric column using IQR or Z-score method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to check for outliers.
        method (str): Detection method — "iqr" or "zscore".
        threshold (float): IQR multiplier (default 1.5) or Z-score cutoff (default 3.0).

    Returns:
        pd.Series: Boolean mask where True indicates an outlier row.
    """
    if column not in df.columns:
        logging.error(f"Column '{column}' not found for outlier detection.")
        return pd.Series([False] * len(df), index=df.index)

    if method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (df[column] < lower) | (df[column] > upper)

    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        if std == 0:
            mask = pd.Series([False] * len(df), index=df.index)
        else:
            z_scores = (df[column] - mean) / std
            mask = z_scores.abs() > threshold

    else:
        logging.warning(f"Unknown outlier method '{method}'. Returning no outliers.")
        mask = pd.Series([False] * len(df), index=df.index)

    outlier_count = mask.sum()
    logging.info(f"Outlier detection ({method}): {outlier_count} outlier(s) in '{column}'.")
    return mask


def clean_sales_data(df, date_column="date", missing_strategy="interpolate"):
    """
    Run the full cleaning pipeline on a sales DataFrame.

    Steps:
        1. Remove duplicate rows.
        2. Convert and sort by date.
        3. Handle missing values.

    Args:
        df (pd.DataFrame): Raw sales DataFrame.
        date_column (str): Name of the date column.
        missing_strategy (str): Strategy for handling missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for feature engineering.
    """
    if df is None or df.empty:
        logging.warning("clean_sales_data received an empty DataFrame.")
        return df

    logging.info(f"Starting cleaning pipeline ({len(df)} rows)...")

    df = remove_duplicates(df, subset=[date_column, "product_id"])
    df = convert_date_column(df, date_column=date_column)
    df = handle_missing_values(df, strategy=missing_strategy)

    logging.info(f"Cleaning pipeline complete ({len(df)} rows).")
    return df
