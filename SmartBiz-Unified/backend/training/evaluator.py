"""
evaluator.py
Computes MAE, RMSE, WAPE, sMAPE and saves a model comparison CSV.
"""

import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — handles zero actuals gracefully."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float('nan')
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200 scale)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return float('nan')
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return {
        'MAE':   mae(y_true, y_pred),
        'RMSE':  rmse(y_true, y_pred),
        'WAPE':  wape(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
    }


class Evaluator:
    """
    Collects model results, computes metrics, prints a comparison table,
    and saves the results CSV to experiments/evaluation_results/.
    """

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def evaluate_all(self, all_results: dict) -> pd.DataFrame:
        """
        all_results: dict[model_name] -> {'y_true': array, 'y_pred': array}
        Returns a DataFrame: Model | MAE | RMSE | WAPE | sMAPE
        """
        rows = []
        for model_name, data in all_results.items():
            metrics = compute_all(data['y_true'], data['y_pred'])
            row = {'Model': model_name, **metrics}
            rows.append(row)
            print(f"  {model_name:<20}  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}"
                  f"  WAPE={metrics['WAPE']:.4f}  sMAPE={metrics['sMAPE']:.2f}%")

        comparison_df = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
        save_path = os.path.join(self.results_dir, 'model_comparison.csv')
        comparison_df.to_csv(save_path, index=False)
        print(f"\n[Evaluator] Comparison table saved -> {save_path}")
        return comparison_df
