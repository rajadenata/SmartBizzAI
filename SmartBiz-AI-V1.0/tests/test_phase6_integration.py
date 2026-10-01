import unittest
import pandas as pd
import os
import json
import shutil
from src.pipeline.smartbiz_pipeline import SmartBizPipeline
from src.pipeline.pipeline_runner import run_pipeline

class TestPhase6Integration(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        dates = pd.date_range(start="2026-01-01", periods=50, freq='D')
        self.df = pd.DataFrame({
            "date": dates,
            "product_id": ["P001"] * 50,
            "sales_qty": [10 + (i % 5) for i in range(50)],
            "promotion": [0] * 50
        })
        self.results_dir = "tests/temp_integration_results"
        
    def tearDown(self):
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
            
    def test_pipeline_instantiation(self):
        pipeline = SmartBizPipeline(model_name="Random Forest")
        self.assertEqual(pipeline.model_name, "Random Forest")
        self.assertIsNotNone(pipeline.forecaster)
        self.assertIsNotNone(pipeline.recommendation_engine)
        
    def test_pipeline_methods(self):
        pipeline = SmartBizPipeline(model_name="LightGBM")
        # 1. Test data processing
        df_product = pipeline.process_data(self.df, "P001")
        self.assertFalse(df_product.empty)
        self.assertIn("rolling_mean_7", df_product.columns)
        
        # 2. Test train and predict
        predictions, actuals, eval_metrics = pipeline.train_and_predict(df_product)
        self.assertTrue(len(predictions) > 0)
        self.assertIn("wape", eval_metrics)
        self.assertIn("rmse", eval_metrics)
        
        # 3. Test recommendation generation
        rec = pipeline.generate_recommendations(
            product_id="P001",
            forecast_demand=15.0,
            current_stock=5.0,
            production_capacity=50.0,
            eval_metrics=eval_metrics
        )
        self.assertEqual(rec["product"], "P001")
        self.assertIn("recommended_production", rec)
        self.assertIn("status", rec)
        self.assertIn("reason", rec)
        
    def test_pipeline_runner(self):
        # Test full runner and unified output
        results = run_pipeline(
            df=self.df,
            product_id="P001",
            current_stock=5.0,
            production_capacity=50.0,
            model_name="XGBoost",
            results_dir=self.results_dir
        )
        
        # Validate returned object
        self.assertIn("input_info", results)
        self.assertIn("forecast_result", results)
        self.assertIn("evaluation_metrics", results)
        self.assertIn("decision_recommendation", results)
        self.assertEqual(results["model_name"], "XGBoost")
        
        # Validate file output
        json_path = os.path.join(self.results_dir, "smartbiz_prediction_results.json")
        csv_path = os.path.join(self.results_dir, "smartbiz_recommendations.csv")
        
        self.assertTrue(os.path.exists(json_path), "JSON output file was not created.")
        self.assertTrue(os.path.exists(csv_path), "CSV output file was not created.")
        
        with open(json_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["model_name"], "XGBoost")
            self.assertEqual(data["input_info"]["product_id"], "P001")

if __name__ == "__main__":
    unittest.main()
