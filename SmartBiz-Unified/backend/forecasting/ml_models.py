"""
Machine learning forecasting models for SmartBiz AI V1.0.
Implements Random Forest, XGBoost, and LightGBM regressors
with a unified interface for time-series demand forecasting.
"""

import pandas as pd
import numpy as np
import logging

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


class MLModelWrapper:
    """
    Base wrapper class for Machine Learning models.
    Provides a consistent fit, predict, and feature_importance interface.
    """
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.feature_names = None

    def fit(self, X, y):
        """
        Fit the machine learning model.
        
        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (array-like): Training target values.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        logging.info(f"{self.name} fitted successfully.")

    def predict(self, X):
        """
        Predict using the fitted model.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features for prediction.
            
        Returns:
            np.ndarray: Predicted values.
        """
        predictions = self.model.predict(X)
        return np.maximum(0, predictions)  # Demand cannot be negative

    def get_feature_importance(self):
        """
        Retrieve feature importances from the model.
        
        Returns:
            pd.DataFrame: DataFrame containing feature names and their importance scores.
        """
        if not hasattr(self.model, "feature_importances_"):
            logging.warning(f"{self.name} does not support feature importances.")
            return None

        importances = self.model.feature_importances_
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(importances))]
            
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        
        return importance_df


class RandomForestForecaster(MLModelWrapper):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state, 
            **kwargs
        )
        super().__init__(model=model, name="Random Forest")


class XGBoostForecaster(MLModelWrapper):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, **kwargs):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="reg:squarederror",
            **kwargs
        )
        super().__init__(model=model, name="XGBoost")


class LightGBMForecaster(MLModelWrapper):
    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1, random_state=42, **kwargs):
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
            **kwargs
        )
        super().__init__(model=model, name="LightGBM")
