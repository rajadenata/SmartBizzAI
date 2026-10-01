import sqlite3
import logging
import os
import sys

# Ensure config can be imported if run standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.settings import DATABASE_PATH

def create_connection():
    """
    Connect to the SQLite database and return the connection object.
    Creates the database file if it does not exist.
    """
    conn = None
    try:
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DATABASE_PATH)
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
    return conn
