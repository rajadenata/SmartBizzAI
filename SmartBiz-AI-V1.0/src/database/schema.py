"""
Database schema definition for SmartBiz AI V1.0.
Handles table creation for all core business entities.
"""

import sqlite3
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import create_connection


def initialize_database():
    """
    Initialize the SQLite database by creating all required tables.
    Tables are created only if they do not already exist.
    """
    conn = create_connection()
    if conn is None:
        logging.error("Cannot initialize database: connection failed.")
        return False

    try:
        cursor = conn.cursor()

        # Table: businesses
        # Stores information about each registered culinary MSME business.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS businesses (
                business_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                business_name TEXT    NOT NULL,
                category      TEXT,
                location      TEXT
            );
        """)

        # Table: products
        # Stores all products linked to a business.
        # business_id references businesses to enforce a one-to-many relationship.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id      INTEGER NOT NULL,
                product_name     TEXT    NOT NULL,
                product_category TEXT,
                price            REAL,
                FOREIGN KEY (business_id) REFERENCES businesses (business_id)
            );
        """)

        # Table: sales_history
        # Records each daily sales transaction for a product.
        # promotion flag (0/1) indicates whether a promotion was active on that date.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales_history (
                sales_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT    NOT NULL,
                business_id INTEGER NOT NULL,
                product_id  INTEGER NOT NULL,
                sales_qty   INTEGER NOT NULL,
                price       REAL    NOT NULL,
                promotion   INTEGER DEFAULT 0,
                FOREIGN KEY (business_id) REFERENCES businesses (business_id),
                FOREIGN KEY (product_id)  REFERENCES products  (product_id)
            );
        """)

        # Table: inventory
        # Tracks daily inventory levels for each product.
        # stockout_flag (0/1) indicates whether the product was out of stock.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                inventory_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT    NOT NULL,
                product_id      INTEGER NOT NULL,
                stock_available INTEGER NOT NULL,
                stockout_flag   INTEGER DEFAULT 0,
                FOREIGN KEY (product_id) REFERENCES products (product_id)
            );
        """)

        conn.commit()
        logging.info("Database initialized successfully.")
        return True

    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        return False

    finally:
        conn.close()
