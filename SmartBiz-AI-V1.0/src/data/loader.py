"""
CSV data loader module for SmartBiz AI V1.0.
Handles reading CSV files into Pandas DataFrames.
"""

import pandas as pd
import logging
import os


def load_csv(file_path):
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): Absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data, or None if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded CSV: {file_path} ({len(df)} rows)")
        return df

    except pd.errors.EmptyDataError:
        logging.error(f"CSV file is empty: {file_path}")
        return None

    except Exception as e:
        logging.error(f"Error loading CSV file '{file_path}': {e}")
        return None
