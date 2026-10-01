"""
Runner for the SmartBiz AI Pipeline.
Handles saving of unified output format.
"""
import pandas as pd
import json
import os
import logging
from typing import Dict, Any

from src.pipeline.smartbiz_pipeline import SmartBizPipeline

def run_pipeline(
    df: pd.DataFrame, 
    product_id: str, 
    current_stock: float, 
    production_capacity: float, 
    model_name: str = "Random Forest",
    results_dir: str = "results/integration"
) -> Dict[str, Any]:
    """
    Run the end-to-end pipeline and save unified outputs.
    """
    logging.info(f"Starting pipeline for product {product_id} using {model_name}")
    pipeline = SmartBizPipeline(model_name=model_name)
    
    # 1. Process data
    df_product = pipeline.process_data(df, product_id)
    if df_product.empty:
        logging.warning("No data left after processing.")
        return {}
        
    # 2. Train, Predict, Evaluate
    predictions, actuals, eval_metrics = pipeline.train_and_predict(df_product)
    
    # Take the last prediction as the next day's forecast
    latest_forecast = float(predictions[-1])
    
    # 3. Decision Engine Recommendation
    recommendation = pipeline.generate_recommendations(
        product_id=product_id,
        forecast_demand=latest_forecast,
        current_stock=current_stock,
        production_capacity=production_capacity,
        eval_metrics=eval_metrics
    )
    
    # 4. Create unified output
    unified_output = {
        "input_info": {
            "product_id": product_id,
            "current_stock": current_stock,
            "production_capacity": production_capacity
        },
        "forecast_result": {
            "latest_prediction": latest_forecast
        },
        "model_name": model_name,
        "evaluation_metrics": eval_metrics,
        "decision_recommendation": recommendation
    }
    
    # 5. Save results
    os.makedirs(results_dir, exist_ok=True)
    
    json_path = os.path.join(results_dir, "smartbiz_prediction_results.json")
    with open(json_path, 'w') as f:
        json.dump(unified_output, f, indent=4)
        
    # Create CSV for recommendation
    csv_path = os.path.join(results_dir, "smartbiz_recommendations.csv")
    rec_df = pd.DataFrame([recommendation])
    rec_df.to_csv(csv_path, index=False)
    
    logging.info(f"Results saved to {results_dir}")
    
    return unified_output
