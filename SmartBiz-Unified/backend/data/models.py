"""
Data model definitions for SmartBiz AI V1.0.
Defines expected column structures for each dataset.
"""

# Required columns for a valid sales CSV dataset.
# These columns must be present before any data is loaded into the pipeline.
SALES_REQUIRED_COLUMNS = [
    "date",         # Transaction date (format: YYYY-MM-DD)
    "business_id",  # Unique identifier of the business
    "product_id",   # Unique identifier of the product
    "sales_qty",    # Number of units sold on that date
    "price",        # Selling price per unit
    "promotion",    # Binary flag: 1 if a promotion was active, 0 otherwise
]

# Required columns for a valid inventory CSV dataset.
INVENTORY_REQUIRED_COLUMNS = [
    "date",             # Inventory date (format: YYYY-MM-DD)
    "product_id",       # Unique identifier of the product
    "stock_available",  # Number of units available in stock
    "stockout_flag",    # Binary flag: 1 if stockout occurred, 0 otherwise
]
