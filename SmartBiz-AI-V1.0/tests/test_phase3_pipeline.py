"""
Phase 3 — Comprehensive Test Suite for SmartBiz AI V1.0
Tests preprocessing, model training, and metric calculation.

Run from the project root:
    python tests/test_phase3_pipeline.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def record(test_name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append({"test": test_name, "status": status, "detail": detail})
    print(f"  {status}  {test_name}" + (f"  — {detail}" if detail else ""))


# =========================================================================
# SECTION 1: PREPROCESSING TESTS
# =========================================================================
print("\n" + "=" * 60)
print("SECTION 1: PREPROCESSING PIPELINE TESTS")
print("=" * 60)

# --- 1.1  Load CSV ---
try:
    from src.data.loader import load_csv
    df_raw = load_csv("data/raw/sample_sales_data.csv")
    record("1.1 CSV Loading", df_raw is not None and len(df_raw) == 110,
           f"{len(df_raw)} rows loaded")
except Exception as e:
    record("1.1 CSV Loading", False, str(e))

# --- 1.2  Data Cleaning ---
try:
    from src.preprocessing.cleaning import clean_sales_data
    df_clean = clean_sales_data(df_raw.copy())
    has_datetime = pd.api.types.is_datetime64_any_dtype(df_clean["date"])
    no_missing = df_clean.isnull().sum().sum() == 0
    record("1.2 Data Cleaning", has_datetime and no_missing,
           f"datetime={has_datetime}, missing={df_clean.isnull().sum().sum()}")
except Exception as e:
    record("1.2 Data Cleaning", False, str(e))

# --- 1.3  Missing-Value Handling ---
try:
    from src.preprocessing.cleaning import handle_missing_values, convert_date_column
    df_dirty = df_raw.copy()
    df_dirty.loc[3, "sales_qty"] = np.nan
    df_dirty.loc[7, "price"] = np.nan
    df_dirty = convert_date_column(df_dirty)
    df_fixed = handle_missing_values(df_dirty, strategy="interpolate")
    record("1.3 Missing-Value Handling", df_fixed.isnull().sum().sum() == 0,
           f"remaining NaN={df_fixed.isnull().sum().sum()}")
except Exception as e:
    record("1.3 Missing-Value Handling", False, str(e))

# --- 1.4  Duplicate Removal ---
try:
    from src.preprocessing.cleaning import remove_duplicates
    df_dup = pd.concat([df_raw.head(5), df_raw.head(5)], ignore_index=True)
    df_dedup = remove_duplicates(df_dup)
    record("1.4 Duplicate Removal", len(df_dedup) == 5,
           f"10 → {len(df_dedup)} rows")
except Exception as e:
    record("1.4 Duplicate Removal", False, str(e))

# --- 1.5  Feature Engineering ---
try:
    from src.preprocessing.feature_engineering import engineer_features
    from src.preprocessing.transformation import filter_product

    df_p1 = filter_product(df_clean, product_id=1)
    df_feat = engineer_features(df_p1)
    expected = ["day_of_week", "month", "weekend_flag", "promotion_flag",
                "lag_1", "lag_7", "rolling_mean_7"]
    all_present = all(f in df_feat.columns for f in expected)
    record("1.5 Feature Engineering", all_present,
           f"features={expected}")
except Exception as e:
    record("1.5 Feature Engineering", False, str(e))

# --- 1.6  Train/Test Split ---
try:
    from src.preprocessing.transformation import split_train_test
    train_df, test_df = split_train_test(df_feat, test_ratio=0.2)
    correct_total = len(train_df) + len(test_df) == len(df_feat)
    chronological = train_df["date"].max() <= test_df["date"].min()
    record("1.6 Train/Test Split", correct_total and chronological,
           f"train={len(train_df)}, test={len(test_df)}, chronological={chronological}")
except Exception as e:
    record("1.6 Train/Test Split", False, str(e))


# =========================================================================
# SECTION 2: MODEL TRAINING TESTS
# =========================================================================
print("\n" + "=" * 60)
print("SECTION 2: MODEL TRAINING TESTS")
print("=" * 60)

train_y = train_df["sales_qty"].values
test_y = test_df["sales_qty"].values
horizon = len(test_y)

# --- 2.1  Seasonal Naive ---
try:
    from src.forecasting.baseline import SeasonalNaive
    sn = SeasonalNaive(season_length=7)
    sn.fit(train_y)
    sn_pred = sn.predict(horizon)
    record("2.1 Seasonal Naive", len(sn_pred) == horizon,
           f"predictions={sn_pred}")
except Exception as e:
    record("2.1 Seasonal Naive", False, str(e))

# --- 2.2  Moving Average ---
try:
    from src.forecasting.baseline import MovingAverage
    ma = MovingAverage(window=7)
    ma.fit(train_y)
    ma_pred = ma.predict(horizon)
    record("2.2 Moving Average", len(ma_pred) == horizon,
           f"forecast_value={ma_pred[0]:.2f}")
except Exception as e:
    record("2.2 Moving Average", False, str(e))

# --- 2.3  Exponential Smoothing ---
try:
    from src.forecasting.baseline import ExponentialSmoothing
    es = ExponentialSmoothing(alpha=0.3)
    es.fit(train_y)
    es_pred = es.predict(horizon)
    record("2.3 Exponential Smoothing", len(es_pred) == horizon,
           f"level={es_pred[0]:.2f}")
except Exception as e:
    record("2.3 Exponential Smoothing", False, str(e))

# --- 2.4  Batch Runner ---
try:
    from src.forecasting.baseline import run_baseline_experiment
    preds = run_baseline_experiment(train_y, test_y)
    has_all = all(k in preds for k in ["Seasonal Naive", "Moving Average",
                                        "Exponential Smoothing"])
    record("2.4 Batch Experiment Runner", has_all,
           f"models={list(preds.keys())}")
except Exception as e:
    record("2.4 Batch Experiment Runner", False, str(e))


# =========================================================================
# SECTION 3: METRIC CALCULATION TESTS
# =========================================================================
print("\n" + "=" * 60)
print("SECTION 3: METRIC CALCULATION TESTS")
print("=" * 60)

# --- 3.1  Individual Metrics (known values) ---
try:
    from src.forecasting.evaluation import (
        mean_absolute_error,
        root_mean_squared_error,
        weighted_absolute_percentage_error,
        symmetric_mean_absolute_percentage_error,
    )

    actual = np.array([10, 20, 30, 40, 50])
    predicted = np.array([12, 18, 33, 37, 48])

    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    wape = weighted_absolute_percentage_error(actual, predicted)
    smape = symmetric_mean_absolute_percentage_error(actual, predicted)

    mae_ok = abs(mae - 2.4) < 0.01
    rmse_ok = rmse > 0
    wape_ok = 0 <= wape <= 1
    smape_ok = 0 <= smape <= 200

    record("3.1 MAE Calculation", mae_ok, f"MAE={mae:.4f} (expected ≈ 2.4)")
    record("3.2 RMSE Calculation", rmse_ok, f"RMSE={rmse:.4f}")
    record("3.3 WAPE Calculation", wape_ok, f"WAPE={wape:.4f}")
    record("3.4 sMAPE Calculation", smape_ok, f"sMAPE={smape:.4f}")
except Exception as e:
    record("3.x Metric Calculations", False, str(e))

# --- 3.5  Structured Evaluation Output ---
try:
    from src.forecasting.evaluation import evaluate_forecast
    result = evaluate_forecast("TestModel", actual, predicted)
    has_keys = all(k in result for k in ["model_name", "mae", "rmse", "wape", "smape"])
    correct_name = result["model_name"] == "TestModel"
    record("3.5 Structured Output Format", has_keys and correct_name,
           f"keys={list(result.keys())}")
except Exception as e:
    record("3.5 Structured Output Format", False, str(e))

# --- 3.6  Multi-Model Evaluation ---
try:
    from src.forecasting.evaluation import evaluate_all_models
    all_results = evaluate_all_models(test_y, preds)
    record("3.6 Multi-Model Evaluation", len(all_results) == 3,
           f"evaluated {len(all_results)} models")
except Exception as e:
    record("3.6 Multi-Model Evaluation", False, str(e))


# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

total = len(results)
passed = sum(1 for r in results if r["status"] == PASS)
failed = total - passed

for r in results:
    print(f"  {r['status']}  {r['test']}")

print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")

if failed == 0:
    print("\n🎉  ALL TESTS PASSED — Phase 3 pipeline is fully operational.")
else:
    print(f"\n⚠️  {failed} test(s) FAILED. Review the output above.")

sys.exit(0 if failed == 0 else 1)
