"""
trainer_ml.py
Machine Learning models: Random Forest, XGBoost, LightGBM.
Uses feature-engineered flat dataframe with a global 80/20 time-based split.
"""

import pandas as pd
import numpy as np
import os
import gc
import joblib
import logging
import psutil
import time
from typing import Tuple, Dict, Generator

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Feature columns contract
# ─────────────────────────────────────────
# These must match the output of backend/features/engineering.py
FEATURE_COLS = [
    'day', 'weekday', 'month', 'year',
    'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_mean_28', 'rolling_std_28',
    'sell_price_lag_1',
]
TARGET_COL = 'demand'


def _time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Global time-based split on sorted date column based on UNIQUE DATES.
    NEVER shuffles — preserves temporal order. All observations belonging to the same date must remain in the same split.
    """
    df = df.sort_values('date').reset_index(drop=True)
    unique_dates = np.sort(df['date'].unique())
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]
    
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    return train_df, test_df

def _walk_forward_splits(df: pd.DataFrame, horizon_days: int = 7, n_folds: int = 3) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    PREPARED DESIGN FOR 7-DAY FORECASTING.
    Generates train/test pairs for rolling-origin evaluation.
    Each test set is exactly `horizon_days` long.
    """
    df = df.sort_values('date').reset_index(drop=True)
    dates = df['date'].sort_values().unique()
    
    if len(dates) < horizon_days * n_folds + 1:
        logger.warning("Not enough dates for full walk-forward evaluation.")
        
    for fold in range(n_folds, 0, -1):
        # Calculate cutoff for test set
        test_start_idx = len(dates) - (fold * horizon_days)
        if test_start_idx < 1:
            continue
            
        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_start_idx + horizon_days - 1] if (test_start_idx + horizon_days - 1) < len(dates) else dates[-1]
        
        train_df = df[df['date'] < test_start_date].copy()
        test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()
        yield train_df, test_df


def _prepare_xy(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature columns (will be skipped): {missing}")
    X = df[available].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y, available


# ─────────────────────────────────────────
# MLTrainer class
# ─────────────────────────────────────────

class MLTrainer:
    """
    Trains Random Forest, XGBoost, and LightGBM on feature-engineered M5 data.
    Uses a global 80/20 time-based split.
    Memory-safe: drops train/test splits after each model is trained.
    """

    MODELS = {
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        ),
        'xgboost': XGBRegressor(
            n_estimators=300,
            max_depth=6,
            tree_method='hist',
            random_state=42,
            verbosity=0,
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=300,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        ),
    }

    MODEL_DIRS = {
        'random_forest': 'random_forest',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
    }

    def __init__(self, model_save_dir: str, train_ratio: float = 0.8):
        self.model_save_dir = model_save_dir
        self.train_ratio = train_ratio
        
        # Bounded thread configuration
        SAFE_N_JOBS = min(4, os.cpu_count() or 1)
        self.MODELS = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                n_jobs=SAFE_N_JOBS,
                random_state=42,
            ),
            'xgboost': XGBRegressor(
                n_estimators=300,
                max_depth=6,
                tree_method='hist',
                n_jobs=SAFE_N_JOBS,
                random_state=42,
                verbosity=0,
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=300,
                num_leaves=31,
                n_jobs=SAFE_N_JOBS,
                random_state=42,
                verbose=-1,
            ),
        }
        
        for subdir in self.MODEL_DIRS.values():
            os.makedirs(os.path.join(model_save_dir, subdir), exist_ok=True)

    def train(self, df: pd.DataFrame, report_data: dict = None, dry_run: bool = False) -> Dict[str, Dict]:
        """
        Trains all ML models on the provided feature-engineered dataframe.
        Returns dict keyed by model name with y_true / y_pred arrays.
        """
        print(f"[MLTrainer] Dataset shape: {df.shape}")
        print(f"[MLTrainer] Performing global time-based split (train={self.train_ratio:.0%})...")

        train_df, test_df = _time_split(df, self.train_ratio)
        
        if report_data is not None:
            report_data['train_start_date'] = str(train_df['date'].min().date())
            report_data['train_end_date'] = str(train_df['date'].max().date())
            report_data['test_start_date'] = str(test_df['date'].min().date())
            report_data['test_end_date'] = str(test_df['date'].max().date())
            report_data['number_of_train_dates'] = train_df['date'].nunique()
            report_data['number_of_test_dates'] = test_df['date'].nunique()
            report_data['number_of_train_rows'] = len(train_df)
            report_data['number_of_test_rows'] = len(test_df)

        X_train, y_train, feature_names = _prepare_xy(train_df)
        X_test, y_test, _ = _prepare_xy(test_df)

        print(f"[MLTrainer] Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
        print(f"[MLTrainer] Features used ({len(feature_names)}): {feature_names}")

        del train_df, test_df
        gc.collect()

        results = {}

        for model_name, model in self.MODELS.items():
            # Memory safety check
            mem = psutil.virtual_memory()
            total_ram_gb = mem.total / (1024**3)
            avail_ram_gb = mem.available / (1024**3)
            min_safe_ram = max(4.0, total_ram_gb * 0.25)
            
            if avail_ram_gb < min_safe_ram:
                raise MemoryError(f"Abort training {model_name}: Available RAM ({avail_ram_gb:.2f}GB) is below safety threshold ({min_safe_ram:.2f}GB).")
                
            print(f"\n[MLTrainer] Training {model_name}...")
            
            rss_before = psutil.Process().memory_info().rss / 1e6
            peak_before = report_data.get('peak_process_rss_mb', rss_before)
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            rss_after_fit = psutil.Process().memory_info().rss / 1e6
            
            if report_data is not None:
                report_data.setdefault('training_times', {})[model_name] = training_time
                if 'models' not in report_data:
                    report_data['models'] = {}
                report_data['models'][model_name] = {
                    'rss_before_mb': rss_before,
                    'rss_after_fit_mb': rss_after_fit,
                    'training_time_sec': training_time
                }

            y_pred = model.predict(X_test).astype(np.float32)
            
            rss_after_predict = psutil.Process().memory_info().rss / 1e6
            if report_data is not None:
                report_data['models'][model_name]['rss_after_predict_mb'] = rss_after_predict
                
                # Estimate peak
                current_peak = max(rss_before, rss_after_fit, rss_after_predict)
                if current_peak > report_data.get('peak_process_rss_mb', 0):
                    report_data['peak_process_rss_mb'] = current_peak
                
                report_data['models'][model_name]['peak_rss_mb'] = current_peak
            results[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'feature_names': feature_names,
            }

            # Save model
            if not dry_run:
                save_dir = os.path.join(self.model_save_dir, self.MODEL_DIRS[model_name])
                save_path = os.path.join(save_dir, f'{model_name}.joblib')
                joblib.dump(model, save_path)
                print(f"[MLTrainer] Saved {model_name} -> {save_path}")
            else:
                print(f"[MLTrainer] Dry-run: skipping model saving for {model_name}")

            del model
            gc.collect()

        del X_train, y_train, X_test
        gc.collect()

        return results
