"""
Phase 4 — Comprehensive Test Suite for SmartBiz AI V1.0
Tests advanced machine learning forecasting models and training pipeline.

Run from the project root:
    python tests/test_phase4_pipeline.py
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
# SETUP DUMMY DATA FOR TESTING
# =========================================================================
print("\n" + "=" * 60)
print("SETUP TEST DATA")
print("=" * 60)

try:
    dates = pd.date_range("2024-01-01", periods=100)
    df_test = pd.DataFrame({
        "date": dates,
        "product_id": 1,
        "sales_qty": np.random.poisson(20, 100),
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "promotion_flag": np.random.randint(0, 2, 100)
    })
    
    # Add lag/rolling to match feature pipeline
    df_test["lag_1"] = df_test["sales_qty"].shift(1)
    df_test["lag_7"] = df_test["sales_qty"].shift(7)
    df_test["rolling_mean_7"] = df_test["sales_qty"].rolling(7, min_periods=1).mean()
    
    # Train/Test Split
    from src.preprocessing.transformation import split_train_test
    train_df, test_df = split_train_test(df_test, test_ratio=0.2)
    record("0.0 Data Setup", True, f"Train: {len(train_df)}, Test: {len(test_df)}")
except Exception as e:
    record("0.0 Data Setup", False, str(e))


# =========================================================================
# SECTION 1: ML MODELS INITIALIZATION & FITTING
# =========================================================================
print("\n" + "=" * 60)
print("SECTION 1: ML MODELS TESTING")
print("=" * 60)

try:
    from src.forecasting.training_pipeline import prepare_features_and_target
    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)
    record("1.1 Prepare Features", X_train.shape[1] > 0 and len(y_train) > 0, 
           f"Features: {X_train.shape[1]}, Samples: {len(y_train)}")
except Exception as e:
    record("1.1 Prepare Features", False, str(e))

# Random Forest
try:
    from src.forecasting.ml_models import RandomForestForecaster
    rf = RandomForestForecaster(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_imp = rf.get_feature_importance()
    record("1.2 Random Forest", len(rf_preds) == len(X_test) and rf_imp is not None, 
           f"Predictions: {len(rf_preds)}")
except Exception as e:
    record("1.2 Random Forest", False, str(e))

# XGBoost
try:
    from src.forecasting.ml_models import XGBoostForecaster
    xgb_model = XGBoostForecaster(n_estimators=10, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_imp = xgb_model.get_feature_importance()
    record("1.3 XGBoost", len(xgb_preds) == len(X_test) and xgb_imp is not None, 
           f"Predictions: {len(xgb_preds)}")
except Exception as e:
    record("1.3 XGBoost", False, str(e))

# LightGBM
try:
    from src.forecasting.ml_models import LightGBMForecaster
    lgb_model = LightGBMForecaster(n_estimators=10, random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    lgb_imp = lgb_model.get_feature_importance()
    record("1.4 LightGBM", len(lgb_preds) == len(X_test) and lgb_imp is not None, 
           f"Predictions: {len(lgb_preds)}")
except Exception as e:
    record("1.4 LightGBM", False, str(e))


# =========================================================================
# SECTION 2: TRAINING PIPELINE
# =========================================================================
print("\n" + "=" * 60)
print("SECTION 2: PIPELINE ORCHESTRATION")
print("=" * 60)

try:
    from src.forecasting.training_pipeline import run_ml_experiment
    preds_dict, importances_dict, y_true = run_ml_experiment(train_df, test_df)
    
    models = ["Random Forest", "XGBoost", "LightGBM"]
    has_all = all(m in preds_dict for m in models) and all(m in importances_dict for m in models)
    record("2.1 ML Experiment Runner", has_all, f"Models trained: {list(preds_dict.keys())}")
except Exception as e:
    record("2.1 ML Experiment Runner", False, str(e))

try:
    from src.forecasting.evaluation import evaluate_all_models
    eval_results = evaluate_all_models(y_true, preds_dict)
    has_results = len(eval_results) == 3
    keys_ok = all(k in eval_results[0] for k in ["model_name", "mae", "rmse", "wape", "smape"])
    record("2.2 Multi-Model Evaluation", has_results and keys_ok, f"Evaluated {len(eval_results)} models")
except Exception as e:
    record("2.2 Multi-Model Evaluation", False, str(e))


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
    print("\n🎉  ALL TESTS PASSED — Phase 4 ML models are fully operational.")
else:
    print(f"\n⚠️  {failed} test(s) FAILED. Review the output above.")

sys.exit(0 if failed == 0 else 1)
