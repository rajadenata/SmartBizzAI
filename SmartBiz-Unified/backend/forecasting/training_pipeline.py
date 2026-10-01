"""
Training pipeline module for SmartBiz AI V1.0.
Orchestrates the process of loading processed data, training machine learning models,
generating predictions, and evaluating performance.
"""

import pandas as pd
import logging
import os
import json

from src.preprocessing.transformation import split_train_test
from src.forecasting.evaluation import evaluate_all_models
from src.forecasting.ml_models import RandomForestForecaster, XGBoostForecaster, LightGBMForecaster

def prepare_features_and_target(df, target_column="sales_qty", exclude_cols=None):
    """
    Separate features and target for ML models.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The column to predict.
        exclude_cols (list): Columns to exclude from features (e.g., 'date', 'product_id').
        
    Returns:
        tuple: (X, y)
    """
    if exclude_cols is None:
        exclude_cols = ["date", "product_id", "business_id"]
        
    # Drop rows with NaN values (e.g., resulting from lag or rolling features)
    df_clean = df.dropna().copy()
    
    X = df_clean.drop(columns=[target_column] + [col for col in exclude_cols if col in df_clean.columns])
    y = df_clean[target_column].values
    
    return X, y


def run_ml_experiment(train_df, test_df, target_column="sales_qty", feature_exclude=None):
    """
    Train ML models, generate predictions, and extract feature importances.
    
    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Testing dataframe.
        target_column (str): Target variable to predict.
        feature_exclude (list): Columns to exclude from features.
        
    Returns:
        tuple: (predictions_dict, importance_dict)
    """
    X_train, y_train = prepare_features_and_target(train_df, target_column, feature_exclude)
    X_test, y_test = prepare_features_and_target(test_df, target_column, feature_exclude)
    
    models = {
        "Random Forest": RandomForestForecaster(),
        "XGBoost": XGBoostForecaster(),
        "LightGBM": LightGBMForecaster()
    }
    
    predictions = {}
    importances = {}
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        logging.info(f"Predicting with {name}...")
        predictions[name] = model.predict(X_test)
        
        # Get feature importances
        imp_df = model.get_feature_importance()
        if imp_df is not None:
            importances[name] = imp_df
            
    return predictions, importances, y_test


def save_experiment_results(predictions, importances, test_dates, actual_y, results_dir="results"):
    """
    Save predictions and feature importances to disk.
    
    Args:
        predictions (dict): Model predictions.
        importances (dict): Feature importances per model.
        test_dates (array-like): Test dates for predictions DataFrame.
        actual_y (array-like): Actual test target values.
        results_dir (str): Base directory for saving results.
    """
    # 1. Save Predictions
    pred_dir = os.path.join(results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    pred_df = pd.DataFrame({'date': test_dates, 'actual': actual_y})
    for name, preds in predictions.items():
        pred_df[name] = preds
        
    pred_path = os.path.join(pred_dir, "ml_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logging.info(f"Predictions saved to {pred_path}")
    
    # 2. Save Feature Importances
    feat_dir = os.path.join(results_dir, "feature_importance")
    os.makedirs(feat_dir, exist_ok=True)
    
    for name, imp_df in importances.items():
        filename = f"{name.lower().replace(' ', '_')}_importance.csv"
        path = os.path.join(feat_dir, filename)
        imp_df.to_csv(path, index=False)
    logging.info(f"Feature importances saved to {feat_dir}")
