"""
Forecast evaluation module for SmartBiz AI V1.0.
Implements standard accuracy metrics for demand forecasting:
    - MAE  (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - WAPE (Weighted Absolute Percentage Error)
    - sMAPE (Symmetric Mean Absolute Percentage Error)

All functions accept array-like inputs (lists, numpy arrays, or pandas Series).
"""

import numpy as np
import logging


def _to_array(values):
    """Convert input to a float numpy array."""
    return np.array(values, dtype=float)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def mean_absolute_error(actual, predicted):
    """
    Mean Absolute Error (MAE).

    MAE = (1/n) * Σ |actual_i − predicted_i|

    Args:
        actual (array-like): Ground-truth values.
        predicted (array-like): Forecast values.

    Returns:
        float: MAE value.
    """
    a, p = _to_array(actual), _to_array(predicted)
    return float(np.mean(np.abs(a - p)))


def root_mean_squared_error(actual, predicted):
    """
    Root Mean Squared Error (RMSE).

    RMSE = √[ (1/n) * Σ (actual_i − predicted_i)² ]

    Args:
        actual (array-like): Ground-truth values.
        predicted (array-like): Forecast values.

    Returns:
        float: RMSE value.
    """
    a, p = _to_array(actual), _to_array(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def weighted_absolute_percentage_error(actual, predicted):
    """
    Weighted Absolute Percentage Error (WAPE).

    WAPE = Σ |actual_i − predicted_i| / Σ |actual_i|

    Robust alternative to MAPE — avoids division by zero on individual
    observations by dividing by the total absolute actual values.

    Args:
        actual (array-like): Ground-truth values.
        predicted (array-like): Forecast values.

    Returns:
        float: WAPE value (0.0–∞, lower is better).
    """
    a, p = _to_array(actual), _to_array(predicted)
    total_actual = np.sum(np.abs(a))
    if total_actual == 0:
        logging.warning("WAPE: total actual is zero; returning 0.0.")
        return 0.0
    return float(np.sum(np.abs(a - p)) / total_actual)


def symmetric_mean_absolute_percentage_error(actual, predicted):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE = (100/n) * Σ [ 2·|actual_i − predicted_i| / (|actual_i| + |predicted_i|) ]

    Bounded between 0% and 200%. Handles zeros more gracefully than MAPE.

    Args:
        actual (array-like): Ground-truth values.
        predicted (array-like): Forecast values.

    Returns:
        float: sMAPE value as a percentage (0–200).
    """
    a, p = _to_array(actual), _to_array(predicted)
    denominator = np.abs(a) + np.abs(p)
    # Avoid 0/0: when both actual and predicted are 0, error contribution is 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator == 0, 0.0,
                         2.0 * np.abs(a - p) / denominator)
    return float(100.0 * np.mean(ratio))


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_forecast(model_name, actual, predicted):
    """
    Calculate all four metrics and return a structured result dictionary.

    Args:
        model_name (str): Human-readable model name.
        actual (array-like): Ground-truth values.
        predicted (array-like): Forecast values.

    Returns:
        dict: {
            "model_name": str,
            "mae":   float,
            "rmse":  float,
            "wape":  float,
            "smape": float,
        }
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    wape = weighted_absolute_percentage_error(actual, predicted)
    smape = symmetric_mean_absolute_percentage_error(actual, predicted)

    result = {
        "model_name": model_name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "smape": round(smape, 4),
    }

    logging.info(
        f"[{model_name}]  MAE={result['mae']:.4f}  RMSE={result['rmse']:.4f}  "
        f"WAPE={result['wape']:.4f}  sMAPE={result['smape']:.4f}"
    )
    return result


def evaluate_all_models(actual, predictions_dict):
    """
    Evaluate multiple models and return a list of result dictionaries.

    Args:
        actual (array-like): Ground-truth test values.
        predictions_dict (dict): {model_name: predicted_values}.

    Returns:
        list[dict]: One evaluation dict per model (same format as
            :func:`evaluate_forecast`).
    """
    results = []
    for model_name, predicted in predictions_dict.items():
        result = evaluate_forecast(model_name, actual, predicted)
        results.append(result)
    return results
