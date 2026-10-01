"""
SmartBiz AI V1.0 End-to-End Pipeline
Orchestrates forecasting models and the decision engine.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

from src.preprocessing.cleaning import clean_sales_data
from src.preprocessing.transformation import aggregate_daily_sales, split_train_test, filter_product
from src.preprocessing.feature_engineering import engineer_features
from src.forecasting.training_pipeline import prepare_features_and_target
from src.forecasting.ml_models import RandomForestForecaster, XGBoostForecaster, LightGBMForecaster
from src.forecasting.evaluation import evaluate_forecast
from src.decision_engine.recommendation_engine import RecommendationEngine

class SmartBizPipeline:
    def __init__(self, model_name: str = "Random Forest"):
        self.model_name = model_name
        self.models = {
            "Random Forest": RandomForestForecaster(),
            "XGBoost": XGBoostForecaster(),
            "LightGBM": LightGBMForecaster()
        }
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported.")
            
        self.forecaster = self.models[model_name]
        self.recommendation_engine = RecommendationEngine()
        
    def process_data(self, df: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """Preprocess and engineer features for a specific product."""
        df_clean = clean_sales_data(df)
        df_daily = aggregate_daily_sales(df_clean)
        df_features = engineer_features(df_daily)
        df_product = filter_product(df_features, product_id)
        return df_product
        
    def train_and_predict(self, df_product: pd.DataFrame, target_column: str = "sales_qty", exclude_cols: List[str] = None):
        """Train model, predict and evaluate on test set."""
        train_df, test_df = split_train_test(df_product)
        
        X_train, y_train = prepare_features_and_target(train_df, target_column, exclude_cols)
        X_test, y_test = prepare_features_and_target(test_df, target_column, exclude_cols)
        
        logging.info(f"Training {self.model_name}...")
        self.forecaster.fit(X_train, y_train)
        
        logging.info(f"Predicting with {self.model_name}...")
        predictions = self.forecaster.predict(X_test)
        
        eval_metrics = evaluate_forecast(self.model_name, y_test, predictions)
        
        return predictions, y_test, eval_metrics
        
    def generate_recommendations(
        self, 
        product_id: str, 
        forecast_demand: float, 
        current_stock: float, 
        production_capacity: float, 
        eval_metrics: Dict[str, float],
        safety_percentage: float = 10.0
    ) -> Dict[str, Any]:
        """Generate a production recommendation based on the forecast and its uncertainty."""
        
        # Determine uncertainty based on WAPE
        wape = eval_metrics.get("wape", 0)
        uncertainty = "high" if wape > 0.3 else ("medium" if wape > 0.15 else "low")
        
        rec = self.recommendation_engine.generate_recommendation(
            product_id=product_id,
            forecast_demand=forecast_demand,
            current_stock=current_stock,
            production_capacity=production_capacity,
            safety_percentage=safety_percentage,
            forecast_uncertainty=uncertainty
        )
        return rec
