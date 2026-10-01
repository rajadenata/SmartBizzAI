import unittest
import os
import shutil
from src.decision_engine.decision_rules import DecisionRules
from src.decision_engine.recommendation_engine import RecommendationEngine
from src.decision_engine.production_optimizer import ProductionOptimizer

class TestDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.rules = DecisionRules()
        self.engine = RecommendationEngine()
        self.optimizer = ProductionOptimizer()
        self.output_dir = "results/decision_engine/"

    def test_forecast_higher_than_stock(self):
        """1. Forecast higher than stock -> Expected: Additional production recommendation."""
        # Forecast: 120, Stock: 40, Capacity: 200, Buffer: 10%
        # Target: 120 + 12 = 132. Rec: 132 - 40 = 92
        rec, status, reason = self.rules.apply_rules(120, 40, 200, 10.0)
        self.assertEqual(round(rec), 92)
        self.assertEqual(status, "Increase production")
        self.assertIn("predicted demand is higher", reason)

    def test_stock_higher_than_forecast(self):
        """2. Stock higher than forecast -> Expected: No additional production recommendation."""
        # Forecast: 100, Stock: 150, Buffer: 10% (buffer = 10)
        rec, status, reason = self.rules.apply_rules(100, 150, 200, 10.0)
        self.assertEqual(rec, 0)
        self.assertEqual(status, "No additional production required")

    def test_production_exceeds_capacity(self):
        """3. Production exceeds capacity -> Expected: Recommendation adjusted."""
        # Forecast: 300, Stock: 20, Capacity: 250, Buffer: 10% (30)
        # Rec raw: 300 - 20 + 30 = 310 -> Adjusted to 250
        rec, status, reason = self.rules.apply_rules(300, 20, 250, 10.0)
        self.assertEqual(rec, 250)
        self.assertEqual(status, "Capacity limit reached")

    def test_safety_buffer_calculation(self):
        """4. Safety buffer calculation -> Expected: Correct calculation."""
        # Example from prompt: Forecast 100, stock 20, buffer 10% -> Rec 90
        rec = self.engine.generate_recommendation("Chicken Rice", 100, 20, 500, 10.0)
        self.assertEqual(rec["recommended_production"], 90)
        
        # Test uncertainty high (adds 10% buffer) -> buffer 20%
        # Forecast 100, stock 20, buffer 20% (20). Rec = 100 - 20 + 20 = 100
        rec_high = self.engine.generate_recommendation("Chicken Rice", 100, 20, 500, 10.0, forecast_uncertainty="high")
        self.assertEqual(rec_high["recommended_production"], 100)

    def test_full_decision_pipeline_execution(self):
        """5. Full decision pipeline execution."""
        batch = [
            {"product_id": "P1", "forecast_demand": 120, "current_stock": 30, "production_capacity": 200, "safety_percentage": 10},
            {"product_id": "P2", "forecast_demand": 50, "current_stock": 100, "production_capacity": 100, "safety_percentage": 10},
            {"product_id": "P3", "forecast_demand": 300, "current_stock": 0, "production_capacity": 250, "safety_percentage": 10, "forecast_uncertainty": "high"}
        ]
        results = self.optimizer.optimize_batch(batch)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["recommended_production"], 102) # 120 - 30 + 12
        self.assertEqual(results[1]["recommended_production"], 0)   # Stock > Forecast
        self.assertEqual(results[2]["recommended_production"], 250) # Capacity limited
        
        # Test saving results
        self.optimizer.save_results(results, self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "production_recommendations.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "decision_results.json")))

if __name__ == '__main__':
    unittest.main()
