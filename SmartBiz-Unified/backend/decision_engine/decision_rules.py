class DecisionRules:
    @staticmethod
    def apply_rules(forecast_demand: float, current_stock: float, production_capacity: float, safety_percentage: float, forecast_uncertainty: str = "low"):
        # Rule 4: IF forecast uncertainty is high THEN increase safety buffer
        adjusted_safety_percentage = safety_percentage
        if forecast_uncertainty.lower() == "high":
            adjusted_safety_percentage += 10.0
            
        safety_buffer = forecast_demand * (adjusted_safety_percentage / 100.0)
        
        # Core Formula: Forecast Demand - Available Stock + Safety Buffer Adjustment
        recommended_production = forecast_demand - current_stock + safety_buffer
        
        # Rule 1: IF available_stock >= forecast_demand (and buffer) THEN recommendation = "No additional production required"
        if current_stock >= forecast_demand + safety_buffer:
            return 0, "No additional production required", "Current inventory is sufficient to meet forecast demand and safety buffer."
            
        # Rule 2: IF available_stock < forecast_demand THEN recommend additional production quantity
        status = "Increase production"
        reason = "Production recommendation increased because predicted demand is higher than current inventory availability."
        
        # Rule 3: IF recommendation exceeds production capacity THEN adjust recommendation according to capacity limit
        if recommended_production > production_capacity:
            recommended_production = production_capacity
            status = "Capacity limit reached"
            reason = "Recommendation exceeds production capacity; adjusted to maximum capacity."
            
        return recommended_production, status, reason
