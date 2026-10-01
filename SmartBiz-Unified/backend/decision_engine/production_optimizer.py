import json
import csv
import os
from .recommendation_engine import RecommendationEngine

class ProductionOptimizer:
    def __init__(self):
        self.engine = RecommendationEngine()
        
    def optimize_batch(self, batch_inputs: list) -> list:
        """
        Processes a batch of forecast predictions and generates recommendations.
        """
        results = []
        for item in batch_inputs:
            rec = self.engine.generate_recommendation(
                product_id=item.get("product_id", "Unknown"),
                forecast_demand=item.get("forecast_demand", 0),
                current_stock=item.get("current_stock", 0),
                production_capacity=item.get("production_capacity", float('inf')),
                safety_percentage=item.get("safety_percentage", 10.0),
                forecast_uncertainty=item.get("forecast_uncertainty", "low")
            )
            results.append(rec)
        return results

    def save_results(self, results: list, output_dir: str):
        """
        Saves recommendations to CSV and JSON formats.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, "decision_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
            
        csv_path = os.path.join(output_dir, "production_recommendations.csv")
        if results:
            keys = results[0].keys()
            with open(csv_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
