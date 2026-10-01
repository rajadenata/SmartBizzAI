"""
trainer_baseline.py
Baseline forecasting models: Seasonal Naive, Moving Average, Exponential Smoothing.
These are fit on a per-series (store_id + cat_id) basis and predictions are aggregated.
"""

import pandas as pd
import numpy as np
import os
import gc
import joblib
import logging
import time
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Per-series forecasting functions
# ─────────────────────────────────────────

def seasonal_naive_forecast(train_series: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last seasonal cycle for `horizon` steps."""
    cycle = train_series.values[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(cycle, reps)[:horizon].astype(float)


def moving_average_forecast(train_series: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Use the mean of the last `window` observations for all horizon steps."""
    last_window = train_series.values[-window:]
    avg = np.mean(last_window)
    return np.full(horizon, avg, dtype=float)


def exponential_smoothing_forecast(train_series: pd.Series, horizon: int, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential smoothing (Holt's level-only) forecasted forward."""
    level = float(train_series.iloc[0])
    for val in train_series.iloc[1:]:
        level = alpha * float(val) + (1 - alpha) * level
    return np.full(horizon, level, dtype=float)


# ─────────────────────────────────────────
# BaselineTrainer class
# ─────────────────────────────────────────

class BaselineTrainer:
    """
    Trains three baseline models (Seasonal Naive, Moving Average, Exponential Smoothing)
    on each unique (store_id, cat_id) series using an 80/20 time-based split.
    """

    MODEL_NAMES = ['seasonal_naive', 'moving_average', 'exp_smoothing']

    def __init__(self, model_save_dir: str, train_ratio: float = 0.8):
        self.model_save_dir = os.path.join(model_save_dir, 'baseline')
        self.train_ratio = train_ratio
        os.makedirs(self.model_save_dir, exist_ok=True)

    def _time_split(self, series: pd.DataFrame, split_date) -> Tuple[pd.Series, pd.Series]:
        train_s = series[series['date'] < split_date]['demand']
        test_s = series[series['date'] >= split_date]['demand']
        return train_s, test_s

    def train(self, df: pd.DataFrame, report_data: dict = None, dry_run: bool = False) -> Dict[str, Dict]:
        """
        For each (store_id, cat_id) pair, fit baselines and store train/test predictions.
        Returns: dict keyed by model name, containing list of (y_true, y_pred) arrays.
        """
        start_time = time.time()
        
        unique_dates = np.sort(df['date'].unique())
        split_idx = int(len(unique_dates) * self.train_ratio)
        split_date = unique_dates[split_idx]
        
        series_groups = df.groupby(['store_id', 'cat_id'])
        results = {name: {'y_true': [], 'y_pred': []} for name in self.MODEL_NAMES}
        model_artifacts = {}

        logger.info(f"Training baselines on {series_groups.ngroups} series...")
        print(f"[BaselineTrainer] Training on {series_groups.ngroups} series...")

        for (store, cat), group in series_groups:
            group = group.sort_values('date')
            train_s, test_s = self._time_split(group, split_date)
            horizon = len(test_s)

            if horizon == 0 or len(train_s) < 14:
                logger.warning(f"Skipping {store}/{cat}: insufficient data.")
                continue

            preds = {
                'seasonal_naive': seasonal_naive_forecast(train_s, horizon),
                'moving_average': moving_average_forecast(train_s, horizon),
                'exp_smoothing': exponential_smoothing_forecast(train_s, horizon),
            }

            for name, pred in preds.items():
                results[name]['y_true'].append(test_s.values)
                results[name]['y_pred'].append(pred)

            # Save model artifacts (the train series is enough to reproduce the forecast)
            model_artifacts[(store, cat)] = {
                'train_tail_28': train_s.values[-28:].tolist(),
            }

        # Flatten results
        for name in self.MODEL_NAMES:
            results[name]['y_true'] = np.concatenate(results[name]['y_true'])
            results[name]['y_pred'] = np.concatenate(results[name]['y_pred'])

        # Save artifacts
        if not dry_run:
            save_path = os.path.join(self.model_save_dir, 'baseline_artifacts.joblib')
            joblib.dump(model_artifacts, save_path)
            print(f"[BaselineTrainer] Artifacts saved to {save_path}")
        else:
            print("[BaselineTrainer] Dry-run: skipping baseline_artifacts.joblib saving.")
        
        training_time = time.time() - start_time
        if report_data is not None:
            report_data.setdefault('training_times', {})['baselines_all'] = training_time

        return results
