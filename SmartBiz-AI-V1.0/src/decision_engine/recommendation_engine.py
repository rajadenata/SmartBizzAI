from .decision_rules import DecisionRules

class RecommendationEngine:
    def __init__(self):
        self.rules = DecisionRules()

    def generate_recommendation(
        self, 
        product_id: str, 
        forecast_demand: float, 
        current_stock: float, 
        production_capacity: float, 
        safety_percentage: float = 10.0, 
        forecast_uncertainty: str = "low"
    ) -> dict:
        """
        Generate a production recommendation based on forecast and operational constraints.
        """
        recommended_prod, status, reason = self.rules.apply_rules(
            forecast_demand=forecast_demand,
            current_stock=current_stock,
            production_capacity=production_capacity,
            safety_percentage=safety_percentage,
            forecast_uncertainty=forecast_uncertainty
        )
        
        return {
            "product": product_id,
            "forecast_demand": round(forecast_demand, 2),
            "current_stock": round(current_stock, 2),
            "production_capacity": round(production_capacity, 2),
            "safety_percentage": safety_percentage,
            "forecast_uncertainty": forecast_uncertainty,
            "recommended_production": max(0, round(recommended_prod)),
            "status": status,
            "reason": reason
        }
