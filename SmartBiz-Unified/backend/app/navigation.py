import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def render_navigation():
    """
    Renders the Streamlit navigation structure.
    Returns the selected page name.
    """
    st.sidebar.title("Navigation")
    pages = [
        "Dashboard",
        "Data Management",
        "Forecast",
        "Recommendation",
        "AI Assistant"
    ]
    selection = st.sidebar.radio("Go to", pages)
    return selection


def render_placeholder(page_name):
    """
    Renders a placeholder notice for pages not yet implemented.
    """
    st.title(page_name)
    st.info("This module will be implemented in future phases.")


def render_data_management():
    """
    Renders the Data Management page with dataset preview and validation results.
    """
    st.title("Data Management")
    st.markdown("---")

    SAMPLE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../data/raw/sample_sales_data.csv'
    )

    from src.services.data_service import import_sales_csv

    result = import_sales_csv(SAMPLE_PATH)

    # --- Dataset Information ---
    st.subheader("Dataset Information")
    if result["data"] is not None:
        df = result["data"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Date Range", f"{df['date'].min()} → {df['date'].max()}")
    else:
        st.warning("No dataset could be loaded.")

    st.markdown("---")

    # --- Dataset Preview ---
    st.subheader("Sample Dataset Preview")
    if result["data"] is not None:
        st.dataframe(result["data"].head(10), use_container_width=True)
    else:
        st.error("Dataset could not be displayed.")

    st.markdown("---")

    # --- Validation Result ---
    st.subheader("Validation Result")
    validation = result["validation"]
    if validation["status"]:
        st.success("✅ Dataset passed all validation checks.")
    else:
        st.error("❌ Validation failed. Please review the errors below:")
        for err in validation["errors"]:
            st.warning(f"• {err}")
