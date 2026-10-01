# SmartBiz AI V1.0

# Phase 2 Data Management System Documentation

## 1. Phase Objective

The objective of Phase 2 is to build the data infrastructure layer for SmartBiz AI V1.0. This phase creates a structured system for ingesting, validating, and storing business sales data in a SQLite database, forming the reliable data foundation required by the future forecasting and decision engine.

## 2. Implementation Summary

A complete data pipeline has been implemented covering: centralized SQLite schema initialization, CSV ingestion via a dedicated loader module, multi-rule data validation, and a service layer that orchestrates the pipeline. The Streamlit Data Management page now renders live dataset information, a preview table, and validation results.

## 3. Database Schema Design

The SQLite database (`database/smartbiz.db`) includes four tables:

| Table | Purpose |
|---|---|
| `businesses` | Stores registered MSME business entities |
| `products` | Stores product catalog linked to each business |
| `sales_history` | Records daily sales transactions per product |
| `inventory` | Tracks daily stock levels and stockout events |

All tables use `INTEGER PRIMARY KEY AUTOINCREMENT` and enforce foreign key relationships.

## 4. Data Pipeline Architecture

```
Raw Dataset (CSV)
       |
       v
 Data Loader (loader.py)
       |
       v
 Data Validation (validator.py)
       |
       v
 Service Orchestrator (data_service.py)
       |
       v
 SQLite Database (schema.py / connection.py)
       |
       v
 Future Forecasting Engine
```

## 5. New Files Created

- `src/database/schema.py` — Database schema initializer with all four table definitions.
- `src/data/models.py` — Column requirement constants (`SALES_REQUIRED_COLUMNS`, `INVENTORY_REQUIRED_COLUMNS`).
- `src/data/loader.py` — CSV loader using Pandas with file-not-found and empty-file error handling.
- `src/data/validator.py` — Four-check validation: required columns, missing values, date format, numeric types.
- `src/services/data_service.py` — End-to-end pipeline orchestrator returning a structured result dict.
- `data/raw/sample_sales_data.csv` — 110-row realistic culinary MSME sales dataset.
- `docs/development/phase_2_data_management_system.md` — This documentation file.

## 6. Updated Files

- `src/app/navigation.py` — Added `render_data_management()` function with metrics, dataset preview, and validation display.
- `app.py` — Wired database initialization on startup and routed the Data Management page to the new render function.

## 7. Technical Decisions

- **Schema in a dedicated module (`schema.py`):** Isolates DDL from connection logic and service logic for clear separation of concerns.
- **`CREATE TABLE IF NOT EXISTS`:** Ensures idempotent startup — the app can be restarted safely without duplicate table creation errors.
- **Validation returns structured dict:** `{"status": bool, "errors": list}` makes validation results easy to consume in the UI and in automated tests.
- **Service layer (`data_service.py`):** Keeps the UI layer (navigation.py) free of business logic, following MVC-style principles.
- **Sample data designed realistically:** Sales quantities increase during promotion days and vary naturally across weekdays, mirroring real culinary MSME patterns.

## 8. Testing Result

- **Test 1 – Database initialization:** `initialize_database()` runs on app startup and creates all four tables with no errors.
- **Test 2 – CSV load:** `load_csv("data/raw/sample_sales_data.csv")` returns a DataFrame with 110 rows and 6 columns.
- **Test 3 – Validation:** `validate_sales_data(df)` returns `{"status": True, "errors": []}` confirming all checks pass.
- **Test 4 – Streamlit:** `streamlit run app.py` opens successfully; the Data Management page displays metrics, a 10-row preview table, and a green validation success banner.

## 9. Problems and Solutions

- **Problem:** Relative import paths in nested modules (`src/data/validator.py` importing from `src/data/models.py`) can fail depending on the working directory.
- **Solution:** Used module-relative imports (e.g., `from src.data.models import ...`) consistently from the project root, and ensured `sys.path.append` is applied at entry points where needed.

## 10. Completion Checklist

[✓] Database schema created
[✓] SQLite initialized
[✓] Dataset structure defined
[✓] CSV loader created
[✓] Data validation implemented
[✓] Sample dataset created
[✓] Streamlit integration completed
[✓] Testing completed
[✓] Documentation completed

## 11. Phase Status

STATUS:
COMPLETED

READY FOR:
Phase 3 AI Research Pipeline
