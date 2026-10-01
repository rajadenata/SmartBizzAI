"""
Data validation module for SmartBiz AI V1.0.
Checks that loaded DataFrames conform to expected schemas before processing.
"""

import pandas as pd
import logging

from src.data.models import SALES_REQUIRED_COLUMNS


def validate_sales_data(df):
    """
    Validate a sales DataFrame against the required schema.

    Checks performed:
    - Required columns are present.
    - No fully-missing required columns.
    - 'date' column is parseable as a date.
    - Numeric columns contain valid numeric values.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Returns:
        dict: {
            "status": bool,  # True if all checks passed, False otherwise
            "errors": list   # List of error messages encountered
        }
    """
    errors = []

    if df is None:
        return {"status": False, "errors": ["DataFrame is None. Cannot validate."]}

    # --- Check 1: Required columns exist ---
    missing_cols = [col for col in SALES_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # If columns are missing, further checks are not meaningful
    if errors:
        return {"status": False, "errors": errors}

    # --- Check 2: Detect missing values in required columns ---
    for col in SALES_REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} missing value(s).")

    # --- Check 3: Validate date format ---
    try:
        pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    except Exception:
        errors.append("Column 'date' contains values not matching format YYYY-MM-DD.")

    # --- Check 4: Validate numeric columns ---
    numeric_cols = ["sales_qty", "price", "promotion"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric but contains non-numeric values.")

    if errors:
        logging.warning(f"Validation failed with {len(errors)} error(s).")
        return {"status": False, "errors": errors}

    logging.info("Sales data validation passed.")
    return {"status": True, "errors": []}
