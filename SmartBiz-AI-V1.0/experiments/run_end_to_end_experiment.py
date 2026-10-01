"""
End-to-End Experiment script for SmartBiz AI V1.0.
Executes the full pipeline from data loading to decision recommendation.
"""

import pandas as pd
import logging
import os
from src.pipeline.pipeline_runner import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # 1. Load dataset
    data_path = "data/raw/sales_data.csv"
    if os.path.exists(data_path):
        logging.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logging.info(f"Data not found at {data_path}. Generating dummy data for experiment.")
        dates = pd.date_range(start="2026-01-01", periods=100, freq='D')
        df = pd.DataFrame({
            "date": dates,
            "product_id": ["P001"] * 100,
            "sales_qty": [20 + (i % 10) for i in range(100)],
            "promotion": [0] * 100,
            "price": [15000] * 100
        })
        
    product_id = "P001"
    current_stock = 15.0
    production_capacity = 50.0
    
    models = ["Random Forest", "XGBoost", "LightGBM"]
    
    for model in models:
        logging.info(f"--- Running End-to-End Experiment with {model} ---")
        results = run_pipeline(
            df=df,
            product_id=product_id,
            current_stock=current_stock,
            production_capacity=production_capacity,
            model_name=model,
            results_dir=f"results/integration/experiment_{model.replace(' ', '_')}"
        )
        print(f"\nFinal Recommendation Result for {model}:")
        if "decision_recommendation" in results:
            for k, v in results["decision_recommendation"].items():
                print(f"  {k}: {v}")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
