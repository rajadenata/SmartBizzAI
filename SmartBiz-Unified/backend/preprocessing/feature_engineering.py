"""
Feature engineering module for SmartBiz AI V1.0.
Generates temporal, lag, and rolling-window features
designed for daily demand forecasting in culinary MSMEs.
"""

import pandas as pd
import numpy as np
import logging


def add_calendar_features(df, date_column="date"):
    """
    Derive calendar features from the date column.

    Generated features:
        - day_of_week  : 0 (Monday) to 6 (Sunday)
        - month        : 1–12
        - weekend_flag : 1 if Saturday or Sunday, else 0

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column.
        date_column (str): Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with new calendar feature columns.
    """
    if df is None or df.empty:
        logging.warning("add_calendar_features received an empty DataFrame.")
        return df

    df = df.copy()

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])

    df["day_of_week"] = df[date_column].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = df[date_column].dt.month                     # 1–12
    df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)  # Sat/Sun = 1

    logging.info("Calendar features added: day_of_week, month, weekend_flag.")
    return df


def add_promotion_flag(df, promotion_column="promotion"):
    """
    Ensure the promotion flag column exists and is binary (0/1).

    If the column already exists it is coerced to int; otherwise a
    zero-filled column is created.

    Args:
        df (pd.DataFrame): Input DataFrame.
        promotion_column (str): Name of the promotion column.

    Returns:
        pd.DataFrame: DataFrame with a clean promotion_flag column.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if promotion_column in df.columns:
        df["promotion_flag"] = df[promotion_column].fillna(0).astype(int)
    else:
        df["promotion_flag"] = 0
        logging.info("No promotion column found — created promotion_flag with all 0s.")

    logging.info("promotion_flag column ready.")
    return df


def add_lag_features(df, target_column="sales_qty", lags=None):
    """
    Create lag features for the target column.

    Each lag feature contains the target value shifted by N periods back
    in time, capturing autoregressive dependencies.

    Args:
        df (pd.DataFrame): Input DataFrame (must be sorted by date).
        target_column (str): Column to lag.
        lags (list[int] or None): Lag periods. Default is [1, 7].

    Returns:
        pd.DataFrame: DataFrame with new lag columns.
    """
    if df is None or df.empty:
        return df

    if lags is None:
        lags = [1, 7]

    df = df.copy()

    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df[target_column].shift(lag)
        logging.info(f"Lag feature added: {col_name}")

    return df


def add_rolling_features(df, target_column="sales_qty", windows=None):
    """
    Create rolling-window statistics for the target column.

    Computes rolling mean over specified window sizes.  Rolling windows
    use `min_periods=1` so that early rows still receive a value rather
    than NaN.

    Args:
        df (pd.DataFrame): Input DataFrame (must be sorted by date).
        target_column (str): Column to compute rolling stats on.
        windows (list[int] or None): Window sizes. Default is [7].

    Returns:
        pd.DataFrame: DataFrame with new rolling-mean columns.
    """
    if df is None or df.empty:
        return df

    if windows is None:
        windows = [7]

    df = df.copy()

    for window in windows:
        col_name = f"rolling_mean_{window}"
        df[col_name] = (
            df[target_column]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        logging.info(f"Rolling feature added: {col_name}")

    return df


def engineer_features(df, date_column="date", target_column="sales_qty",
                       promotion_column="promotion"):
    """
    Run the full feature-engineering pipeline on a cleaned sales DataFrame.

    Steps:
        1. Add calendar features (day_of_week, month, weekend_flag).
        2. Add promotion_flag.
        3. Add lag features (lag_1, lag_7).
        4. Add rolling mean (rolling_mean_7).

    Args:
        df (pd.DataFrame): Cleaned sales DataFrame.
        date_column (str): Name of the date column.
        target_column (str): Name of the target column.
        promotion_column (str): Name of the raw promotion column.

    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    if df is None or df.empty:
        logging.warning("engineer_features received an empty DataFrame.")
        return df

    logging.info(f"Starting feature engineering ({len(df)} rows)...")

    df = add_calendar_features(df, date_column=date_column)
    df = add_promotion_flag(df, promotion_column=promotion_column)
    df = add_lag_features(df, target_column=target_column, lags=[1, 7])
    df = add_rolling_features(df, target_column=target_column, windows=[7])

    logging.info(
        f"Feature engineering complete. Columns: {list(df.columns)}"
    )
    return df
