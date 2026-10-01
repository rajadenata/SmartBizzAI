"""
Data management service for SmartBiz AI V1.0.
Orchestrates CSV import, validation, and database preparation.
"""

import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import load_csv
from src.data.validator import validate_sales_data


def import_sales_csv(file_path):
    """
    Full pipeline: load a CSV file, validate it, and return a result report.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: {
            "success": bool,
            "data": pd.DataFrame or None,
            "validation": dict,
            "message": str
        }
    """
    # Step 1: Load the CSV
    df = load_csv(file_path)
    if df is None:
        return {
            "success": False,
            "data": None,
            "validation": {"status": False, "errors": ["File could not be loaded."]},
            "message": f"Failed to load file: {file_path}"
        }

    # Step 2: Validate the loaded data
    validation_result = validate_sales_data(df)

    if not validation_result["status"]:
        return {
            "success": False,
            "data": df,
            "validation": validation_result,
            "message": "Validation failed. Please fix the errors before importing."
        }

    # Step 3: Data is clean and ready for future database insertion
    logging.info(f"Data import pipeline successful: {len(df)} rows ready.")
    return {
        "success": True,
        "data": df,
        "validation": validation_result,
        "message": f"Data loaded and validated successfully. {len(df)} rows ready."
    }
