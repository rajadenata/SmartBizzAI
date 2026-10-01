import streamlit as st
import sys
import os

# Add root to path so modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import APP_NAME, APP_VERSION
from src.app.navigation import render_navigation, render_placeholder, render_data_management
from src.database.schema import initialize_database


def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")

    # Initialize database on first startup (safe: only creates tables if missing)
    initialize_database()

    st.sidebar.markdown(f"**{APP_NAME}** v{APP_VERSION}")

    # Render navigation
    page = render_navigation()

    if page == "Dashboard":
        st.title(f"{APP_NAME}")
        st.write("AI-Based Decision Support System for Culinary MSME")
        st.write(f"**Version:** {APP_VERSION}")
    elif page == "Data Management":
        render_data_management()
    else:
        render_placeholder(page)


if __name__ == "__main__":
    main()

