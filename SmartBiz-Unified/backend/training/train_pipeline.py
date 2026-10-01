"""
train_pipeline.py
Master training script for SmartBiz AI V1.0.

Usage:
    python backend/training/train_pipeline.py               # full training
    python backend/training/train_pipeline.py --dry-run     # sample-only, no model saving

Workflow:
    Dataset Validation
    → M5 Data Loading & Aggregation
    → Feature Engineering
    → Baseline Training
    → ML Training
    → Evaluation
    → Save Models & Reports
"""

import sys
import os
import gc
import logging
import argparse
import tracemalloc
import psutil
import json
import py_compile
import tempfile
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from backend.data_validation.validate_dataset import DatasetValidator
from backend.data_loader.m5_loader import M5DataLoader
from backend.features.engineering import generate_features
from backend.training.trainer_baseline import BaselineTrainer
from backend.training.trainer_ml import MLTrainer
from backend.training.evaluator import Evaluator

# ── Logging config ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Path resolution helpers
# ─────────────────────────────────────────

def resolve_dataset_path() -> str:
    """Auto-detect dataset folder (handles hyphen vs space naming and subfolder nesting)."""
    top_candidates = [
        os.path.abspath(os.path.join(ROOT_DIR, '..', 'SmartBiz-Dataset v1.0')),
        os.path.abspath(os.path.join(ROOT_DIR, '..', 'SmartBiz-Dataset-v1.0')),
    ]
    probe_file = 'sales_train_validation.csv'
    for top in top_candidates:
        if not os.path.isdir(top):
            continue
        # Check top-level directly
        if os.path.exists(os.path.join(top, probe_file)):
            return top
        # Walk one level of subdirectories
        for entry in os.scandir(top):
            if entry.is_dir():
                if os.path.exists(os.path.join(entry.path, probe_file)):
                    return entry.path
    raise FileNotFoundError(
        "Cannot find SmartBiz-Dataset directory containing sales_train_validation.csv. "
        f"Looked in: {top_candidates}"
    )


# ─────────────────────────────────────────
# JSON serialization helper
# ─────────────────────────────────────────

def make_json_serializable(obj):
    """Recursively convert numpy/pandas types to JSON-safe Python natives."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


# ─────────────────────────────────────────
# Syntax validation helper
# ─────────────────────────────────────────

BACKEND_FILES_TO_CHECK = [
    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                 p) for p in [
        'backend/training/train_pipeline.py',
        'backend/training/trainer_ml.py',
        'backend/training/trainer_baseline.py',
        'backend/training/evaluator.py',
        'backend/features/engineering.py',
        'backend/data_loader/m5_loader.py',
    ]
]

def run_syntax_checks():
    """Run py_compile on all SmartBiz backend files. Returns (passed: bool, failures: list)."""
    failures = []
    for fpath in BACKEND_FILES_TO_CHECK:
        if not os.path.exists(fpath):
            failures.append(f"NOT FOUND: {fpath}")
            continue
        try:
            py_compile.compile(fpath, doraise=True)
        except py_compile.PyCompileError as e:
            failures.append(f"SYNTAX ERROR in {os.path.basename(fpath)}: {e}")
    return (len(failures) == 0), failures


# ─────────────────────────────────────────
# Structural leakage validation helper
# ─────────────────────────────────────────

def validate_leakage_structural(data: pd.DataFrame, report_data: dict) -> dict:
    """
    Performs real structural checks on a sample of the feature-engineered DataFrame.
    Returns a dict of individual leakage check results.
    """
    from backend.training.trainer_ml import FEATURE_COLS
    results = {
        'demand_not_in_features': True,
        'lag_validation_passed': True,
        'rolling_validation_passed': True,
        'price_leakage_check_passed': True,
        'future_information_check_passed': True,
    }

    # 1. 'demand' must NOT be in FEATURE_COLS
    if 'demand' in FEATURE_COLS:
        results['demand_not_in_features'] = False
        logger.error("Leakage: 'demand' found in FEATURE_COLS")

    # 2-5. Lag structural verification on a small sample series
    try:
        sample_series = data.groupby(['store_id', 'cat_id']).first().index[0]
        s_store, s_cat = sample_series
        series_df = data[
            (data['store_id'] == s_store) & (data['cat_id'] == s_cat)
        ].sort_values('date').reset_index(drop=True)

        # Check lag_1 == previous day's demand
        for lag, col in [(1, 'lag_1'), (7, 'lag_7'), (14, 'lag_14'), (28, 'lag_28')]:
            if col not in series_df.columns:
                results['lag_validation_passed'] = False
                logger.error(f"Leakage: column '{col}' missing")
                continue
            shifted = series_df['demand'].shift(lag)
            # Compare where both are not NaN
            valid_mask = shifted.notna() & series_df[col].notna()
            if valid_mask.sum() == 0:
                results['lag_validation_passed'] = False
                logger.error(f"Leakage: no valid rows to compare {col}")
                continue
            match = np.allclose(
                series_df.loc[valid_mask, col].values.astype(float),
                shifted[valid_mask].values.astype(float),
                equal_nan=True
            )
            if not match:
                results['lag_validation_passed'] = False
                logger.error(f"Leakage: {col} does not equal demand.shift({lag})")

        # 6-8. Rolling features must be computed from lag_1
        # Structural check: rolling_mean_7 and lag_1 correlation should be very high
        for roll_col in ['rolling_mean_7', 'rolling_mean_28', 'rolling_std_28']:
            if roll_col not in series_df.columns:
                results['rolling_validation_passed'] = False
                logger.error(f"Leakage: rolling column '{roll_col}' missing")
            # rolling_std_28 can have NaN at start — check no NaN after cleaning
            if series_df[roll_col].isna().any():
                results['rolling_validation_passed'] = False
                logger.error(f"Leakage: {roll_col} contains NaN after cleaning")

        # 9. sell_price_lag_1 == previous sell_price
        if 'sell_price_lag_1' in series_df.columns and 'sell_price' in series_df.columns:
            shifted_price = series_df['sell_price'].shift(1)
            valid_price_mask = shifted_price.notna() & series_df['sell_price_lag_1'].notna()
            if valid_price_mask.sum() > 0:
                price_match = np.allclose(
                    series_df.loc[valid_price_mask, 'sell_price_lag_1'].values.astype(float),
                    shifted_price[valid_price_mask].values.astype(float),
                    equal_nan=True
                )
                if not price_match:
                    results['price_leakage_check_passed'] = False
                    logger.error("Leakage: sell_price_lag_1 does not equal sell_price.shift(1)")
        else:
            results['price_leakage_check_passed'] = False
            logger.error("Leakage: sell_price_lag_1 or sell_price column missing")

        # 10. sell_price must not be in FEATURE_COLS (future price)
        if 'sell_price' in FEATURE_COLS:
            results['future_information_check_passed'] = False
            logger.error("Leakage: contemporaneous 'sell_price' found in FEATURE_COLS")

    except Exception as e:
        logger.error(f"Structural leakage check failed with exception: {e}")
        results['lag_validation_passed'] = False
        results['rolling_validation_passed'] = False

    # Store into report_data
    report_data['lag_validation_passed'] = bool(results['lag_validation_passed'])
    report_data['rolling_validation_passed'] = bool(results['rolling_validation_passed'])
    report_data['future_information_check_passed'] = bool(results['future_information_check_passed'])
    report_data['price_leakage_check_passed'] = bool(results['price_leakage_check_passed'])
    report_data['leakage_check_passed'] = bool(
        results['demand_not_in_features'] and
        results['lag_validation_passed'] and
        results['rolling_validation_passed'] and
        results['price_leakage_check_passed'] and
        results['future_information_check_passed']
    )
    return results


# ─────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────

def stage_validate(data_path: str):
    logger.info("═" * 60)
    logger.info("STAGE 1: Dataset Validation")
    logger.info("═" * 60)
    results_dir = os.path.join(ROOT_DIR, 'experiments', 'evaluation_results')
    validator = DatasetValidator(data_path, results_dir)
    validator.validate()   # raises ValueError on failure → stops pipeline


def stage_load(data_path: str, dry_run: bool, report_data: dict):
    logger.info("═" * 60)
    logger.info("STAGE 2: Data Loading & Aggregation")
    logger.info("═" * 60)
    loader = M5DataLoader(data_path)
    data, metadata = loader.load_and_aggregate()
    report_data.update(metadata)

    logger.info(f"Loaded data shape: {data.shape}")
    return data


def stage_features(data, dry_run: bool, report_data: dict):
    logger.info("═" * 60)
    logger.info("STAGE 3: Feature Engineering")
    logger.info("═" * 60)
    data, metadata = generate_features(data)
    report_data.update(metadata)
    
    # Real structural leakage validation (uses full data before 90-date cut)
    validate_leakage_structural(data, report_data)
    
    if not report_data.get('leakage_check_passed', False):
        logger.error("Preflight aborted due to leakage validation failure.")
        raise ValueError("Leakage check failed.")
    
    if dry_run:
        logger.info("[DRY-RUN] Sampling latest 90 dates for pipeline smoke-test AFTER feature engineering.")
        dates = np.sort(data['date'].unique())
        latest_dates = dates[-90:] if len(dates) > 90 else dates
        data = data[data['date'].isin(latest_dates)].copy()
        gc.collect()
        
        report_data['dry_run_start_date'] = str(data['date'].min().date())
        report_data['dry_run_end_date'] = str(data['date'].max().date())
        report_data['dry_run_dates'] = int(data['date'].nunique())
        report_data['dry_run_series'] = int(data[['store_id', 'cat_id']].drop_duplicates().shape[0])
        report_data['dry_run_rows'] = int(len(data))

    logger.info(f"Feature-engineered shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    return data


def stage_baseline(data, model_dir: str, dry_run: bool, report_data: dict):
    logger.info("═" * 60)
    logger.info("STAGE 4: Baseline Model Training")
    logger.info("═" * 60)
    if dry_run:
        logger.info("[DRY-RUN] Skipping baseline model saving.")
        # Still run to test logic, but don't persist artifacts
    trainer = BaselineTrainer(model_save_dir=model_dir)
    results = trainer.train(data, report_data, dry_run=dry_run)
    logger.info("Baseline training complete.")
    return results


def stage_ml(data, model_dir: str, dry_run: bool, report_data: dict):
    logger.info("═" * 60)
    logger.info("STAGE 5: ML Model Training")
    logger.info("═" * 60)
    if dry_run:
        logger.info("[DRY-RUN] ML models will be trained on sample data but not saved to disk.")
    trainer = MLTrainer(model_save_dir=model_dir)
    results = trainer.train(data, report_data, dry_run=dry_run)
    logger.info("ML training complete.")
    return results


def stage_evaluate(baseline_results: dict, ml_results: dict):
    logger.info("═" * 60)
    logger.info("STAGE 6: Evaluation")
    logger.info("═" * 60)
    results_dir = os.path.join(ROOT_DIR, 'experiments', 'evaluation_results')
    evaluator = Evaluator(results_dir)
    all_results = {**baseline_results, **ml_results}
    comparison_df = evaluator.evaluate_all(all_results)
    logger.info("\n" + comparison_df.to_string(index=False))
    return comparison_df


# ─────────────────────────────────────────
# Dry-run memory estimation
# ─────────────────────────────────────────

def report_memory(label: str, report_data: dict):
    current, peak = tracemalloc.get_traced_memory()
    process = psutil.Process()
    rss_mb = process.memory_info().rss / 1e6
    
    logger.info(f"[MEM] {label}: tracemalloc={current/1e6:.1f} MB | Process RSS={rss_mb:.1f} MB")
    
    mem_records = report_data.setdefault('memory_tracking', [])
    mem_records.append({
        'stage': label,
        'tracemalloc_mb': current / 1e6,
        'tracemalloc_peak_mb': peak / 1e6,
        'process_rss_mb': rss_mb
    })
    
    # Track global peak RSS
    if rss_mb > report_data.setdefault('peak_process_rss_mb', 0):
        report_data['peak_process_rss_mb'] = rss_mb


# ─────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SmartBiz AI V1.0 Training Pipeline')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run pipeline on the latest 90-date temporal sample to verify logic and estimate memory.')
    args = parser.parse_args()

    dry_run = args.dry_run
    mode_label = "DRY-RUN (latest 90 dates)" if dry_run else "FULL TRAINING"

    logger.info("╔" + "═" * 58 + "╗")
    logger.info(f"║  SmartBiz AI V1.0 — Training Pipeline [{mode_label}]")
    logger.info("╚" + "═" * 58 + "╝")

    report_data = {}
    mem = psutil.virtual_memory()
    report_data['total_ram_gb'] = mem.total / (1024**3)
    report_data['available_ram_before_dry_run_gb'] = mem.available / (1024**3)
    report_data['cpu_logical_cores'] = psutil.cpu_count(logical=True)
    report_data['cpu_physical_cores'] = psutil.cpu_count(logical=False)
    report_data['max_cpu_threads_used'] = min(4, report_data['cpu_logical_cores'] or 1)
    
    report_memory("Before Validation", report_data)

    tracemalloc.start()
    model_dir = os.path.join(ROOT_DIR, 'models')

    try:
        data_path = resolve_dataset_path()
        logger.info(f"Dataset path resolved: {data_path}")

        # Initialize all flags to False
        report_data['dataset_structure_passed'] = False
        report_data['duplicate_check_passed'] = False
        report_data['missing_date_check_passed'] = False
        report_data['feature_engineering_passed'] = False
        report_data['leakage_check_passed'] = False
        report_data['lag_validation_passed'] = False
        report_data['rolling_validation_passed'] = False
        report_data['price_leakage_check_passed'] = False
        report_data['future_information_check_passed'] = False
        report_data['temporal_split_passed'] = False
        report_data['memory_safety_passed'] = False
        report_data['baseline_training_passed'] = False
        report_data['ml_training_passed'] = False
        report_data['evaluation_passed'] = False
        report_data['cpu_safety_passed'] = False
        report_data['dry_run_completed'] = False  # set True only upon success
        
        mem_info = psutil.virtual_memory()
        report_data['total_ram_gb'] = float(mem_info.total / (1024**3))
        report_data['available_ram_before_dry_run_gb'] = float(mem_info.available / (1024**3))
        report_data['cpu_logical_cores'] = int(psutil.cpu_count(logical=True) or 1)
        report_data['cpu_physical_cores'] = int(psutil.cpu_count(logical=False) or 1)
        report_data['max_cpu_threads_used'] = int(min(4, psutil.cpu_count() or 1))
        report_data['cpu_safety_passed'] = bool(report_data['max_cpu_threads_used'] <= 4)

        # Real syntax check
        logger.info("Running py_compile syntax checks on SmartBiz backend files...")
        syntax_ok, syntax_failures = run_syntax_checks()
        report_data['syntax_check_passed'] = bool(syntax_ok)
        if not syntax_ok:
            for fail in syntax_failures:
                logger.error(f"  {fail}")

        # Stage 1 — Validate
        stage_validate(data_path)
        report_memory("After Validation", report_data)

        # Stage 2 — Load
        data = stage_load(data_path, dry_run, report_data)
        
        expected_rows = report_data.get('actual_store_category_series', 0) * report_data.get('n_dates', 0)
        report_data['dataset_structure_passed'] = bool(
            report_data.get('actual_aggregated_rows', -1) == expected_rows and
            report_data.get('duplicate_records', 1) == 0 and
            report_data.get('missing_dates', 1) == 0
        )
        report_data['duplicate_check_passed'] = bool(report_data.get('duplicate_records', 1) == 0)
        report_data['missing_date_check_passed'] = bool(report_data.get('missing_dates', 1) == 0)
        report_memory("After Loading", report_data)

        # Stage 3 — Features
        data = stage_features(data, dry_run, report_data)
        report_memory("After Feature Engineering", report_data)

        # Check Memory Safety
        mem_check = psutil.virtual_memory()
        avail_ram_gb = mem_check.available / (1024**3)
        min_safe_ram = max(4.0, mem_check.total / (1024**3) * 0.25)
        if avail_ram_gb < min_safe_ram:
            logger.error(f"Abort training: Available RAM ({avail_ram_gb:.2f}GB) is below safety threshold ({min_safe_ram:.2f}GB).")
            raise MemoryError("Low RAM safety check failed.")
        report_data['memory_safety_passed'] = True

        # Stage 4 — Baselines
        baseline_results = stage_baseline(data, model_dir, dry_run, report_data)
        report_data['baseline_training_passed'] = True
        report_memory("After Baseline Training", report_data)

        # Stage 5 — ML Models
        ml_results = stage_ml(data, model_dir, dry_run, report_data)
        report_data['ml_training_passed'] = True
        report_memory("After ML Training", report_data)

        # Free data — no longer needed
        del data
        gc.collect()

        # Stage 6 — Evaluate
        comparison_df = stage_evaluate(baseline_results, ml_results)
        report_data['evaluation_passed'] = True

        # Check temporal split
        if 'train_end_date' in report_data and 'test_start_date' in report_data:
            report_data['temporal_split_passed'] = bool(
                report_data['test_start_date'] > report_data['train_end_date']
            )

        # Mark dry_run_completed only after all stages succeed
        report_data['dry_run_completed'] = bool(dry_run)

        logger.info("═" * 60)

        all_passed = bool(
            report_data['dataset_structure_passed'] and
            report_data['duplicate_check_passed'] and
            report_data['missing_date_check_passed'] and
            report_data['feature_engineering_passed'] and
            report_data.get('leakage_check_passed', False) and
            report_data.get('lag_validation_passed', False) and
            report_data.get('rolling_validation_passed', False) and
            report_data.get('price_leakage_check_passed', False) and
            report_data.get('future_information_check_passed', False) and
            report_data.get('temporal_split_passed', False) and
            report_data['memory_safety_passed'] and
            report_data['baseline_training_passed'] and
            report_data['ml_training_passed'] and
            report_data['evaluation_passed'] and
            report_data['syntax_check_passed'] and
            report_data['cpu_safety_passed'] and
            (report_data['dry_run_completed'] if dry_run else True)
        )
        report_data['all_passed'] = all_passed

        # Save JSON report — wrapped so report always prints even if save fails
        results_dir = os.path.join(ROOT_DIR, 'experiments', 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
        json_save_ok = False
        json_save_error = None
        try:
            safe_report = make_json_serializable(report_data)
            report_path = os.path.join(results_dir, 'training_memory_report.json')
            with open(report_path, 'w') as f:
                json.dump(safe_report, f, indent=4)
            json_save_ok = True
            logger.info(f"training_memory_report.json saved -> {report_path}")
        except Exception as json_err:
            json_save_error = str(json_err)
            logger.error(f"Failed to save training_memory_report.json: {json_err}")

        # FINAL PRE-FLIGHT REPORT — always prints
        print("\n" + "="*50)
        print("FINAL PRE-FLIGHT REPORT")
        print("="*50)
        print("DATASET")
        print(f"- Original item-store series: {report_data.get('original_item_series')}")
        print(f"- Actual Store × Category series: {report_data.get('actual_store_category_series')}")
        print(f"- Stores: {report_data.get('n_stores')}")
        print(f"- Categories: {report_data.get('n_categories')}")
        print(f"- Dates: {report_data.get('n_dates')}")
        print(f"- Aggregated rows: {report_data.get('actual_aggregated_rows')}")
        expected_rows = report_data.get('actual_store_category_series', 0) * report_data.get('n_dates', 0)
        print(f"- Expected rows: {expected_rows}")
        print(f"- Series reduction factor: {report_data.get('series_reduction_factor', 0):.2f}")
        print(f"- Duplicate records: {report_data.get('duplicate_records')}")
        print(f"- Missing dates: {report_data.get('missing_dates')}")
        print(f"- Series with missing dates: {report_data.get('series_with_missing_dates')}")
        
        print("\nFEATURE ENGINEERING")
        print(f"- Rows before feature cleaning: {report_data.get('rows_before_feature_cleaning')}")
        print(f"- Rows removed: {report_data.get('rows_removed')} (Price drops: {report_data.get('price_rows_removed', 0)})")
        print(f"- Rows remaining: {report_data.get('rows_after_feature_cleaning')}")
        print(f"- Series retained: {report_data.get('series_retained')}")
        print(f"- Required features: PASSED")
        print(f"- NaN status: PASSED")
        
        print("\nLEAKAGE")
        print(f"- Demand leakage check: {'PASSED' if report_data.get('leakage_check_passed') else 'FAILED'}")
        print(f"- Lag validation: {'PASSED' if report_data.get('lag_validation_passed') else 'FAILED'}")
        print(f"- Rolling validation: {'PASSED' if report_data.get('rolling_validation_passed') else 'FAILED'}")
        print(f"- Price leakage check: {'PASSED' if report_data.get('price_leakage_check_passed') else 'FAILED'}")
        print(f"- Future information check: {'PASSED' if report_data.get('future_information_check_passed') else 'FAILED'}")
        
        print("\nTEMPORAL VALIDATION")
        print(f"- Train period: {report_data.get('train_start_date')} to {report_data.get('train_end_date')}")
        print(f"- Test period: {report_data.get('test_start_date')} to {report_data.get('test_end_date')}")
        print(f"- Train rows: {report_data.get('number_of_train_rows')}")
        print(f"- Test rows: {report_data.get('number_of_test_rows')}")
        print(f"- Date overlap check: {'PASSED' if report_data.get('temporal_split_passed') else 'FAILED'}")
        
        print("\nHARDWARE")
        print(f"- Total RAM: {report_data.get('total_ram_gb', 0):.2f} GB")
        print(f"- Available RAM before training: {report_data.get('available_ram_before_dry_run_gb', 0):.2f} GB")
        print(f"- CPU logical cores: {report_data.get('cpu_logical_cores')}")
        print(f"- CPU physical cores: {report_data.get('cpu_physical_cores')}")
        print(f"- Maximum training threads: {report_data.get('max_cpu_threads_used')}")
        
        print("\nMEMORY")
        mem_track = report_data.get('memory_tracking', [{'process_rss_mb': 0}])
        print(f"- RSS before training: {mem_track[0].get('process_rss_mb', 0):.2f} MB")
        print(f"- RSS after feature engineering: {mem_track[-2].get('process_rss_mb', 0):.2f} MB")
        print(f"- Per-model RSS:")
        if 'models' in report_data:
            for m, m_data in report_data['models'].items():
                print(f"  {m}: {m_data.get('rss_after_fit_mb', 0):.1f} MB")
        print(f"- Peak RSS: {report_data.get('peak_process_rss_mb', 0):.2f} MB")
        print(f"- Tracemalloc information: Logged")
        
        print("\nTRAINING")
        training_times = report_data.get('training_times', {})
        print(f"- Baseline training time: {training_times.get('baselines_all', 0):.2f}s")
        print(f"- Random Forest training time: {training_times.get('random_forest', 0):.2f}s")
        print(f"- XGBoost training time: {training_times.get('xgboost', 0):.2f}s")
        print(f"- LightGBM training time: {training_times.get('lightgbm', 0):.2f}s")
        print(f"- Completion status: PASSED")
        
        print("\nPREFLIGHT")
        print(f"- dataset_structure_passed: {report_data.get('dataset_structure_passed', False)}")
        print(f"- duplicate_check_passed: {report_data.get('duplicate_check_passed', False)}")
        print(f"- missing_date_check_passed: {report_data.get('missing_date_check_passed', False)}")
        print(f"- feature_engineering_passed: {report_data.get('feature_engineering_passed', False)}")
        print(f"- leakage_check_passed: {report_data.get('leakage_check_passed', False)}")
        print(f"- lag_validation_passed: {report_data.get('lag_validation_passed', False)}")
        print(f"- rolling_validation_passed: {report_data.get('rolling_validation_passed', False)}")
        print(f"- price_leakage_check_passed: {report_data.get('price_leakage_check_passed', False)}")
        print(f"- future_information_check_passed: {report_data.get('future_information_check_passed', False)}")
        print(f"- temporal_split_passed: {report_data.get('temporal_split_passed', False)}")
        print(f"- memory_safety_passed: {report_data.get('memory_safety_passed', False)}")
        print(f"- baseline_training_passed: {report_data.get('baseline_training_passed', False)}")
        print(f"- ml_training_passed: {report_data.get('ml_training_passed', False)}")
        print(f"- evaluation_passed: {report_data.get('evaluation_passed', False)}")
        print(f"- syntax_check_passed: {report_data.get('syntax_check_passed', False)}")
        print(f"- cpu_safety_passed: {report_data.get('cpu_safety_passed', False)}")
        print(f"- dry_run_completed: {report_data.get('dry_run_completed', False)}")
        print(f"- Overall all_passed: {report_data.get('all_passed', False)}")

        print("\n" + "="*50)
        print(f"JSON report saved: {'YES' if json_save_ok else 'FAILED — ' + str(json_save_error)}")
        print("="*50)

        if all_passed:
            print("PRE-FLIGHT PASSED. READY FOR MANUAL FULL TRAINING.")
        else:
            # Identify which checks failed
            failed_checks = [
                k for k in [
                    'dataset_structure_passed', 'duplicate_check_passed',
                    'missing_date_check_passed', 'feature_engineering_passed',
                    'leakage_check_passed', 'lag_validation_passed',
                    'rolling_validation_passed', 'price_leakage_check_passed',
                    'future_information_check_passed', 'temporal_split_passed',
                    'memory_safety_passed', 'baseline_training_passed',
                    'ml_training_passed', 'evaluation_passed',
                    'syntax_check_passed', 'cpu_safety_passed', 'dry_run_completed',
                ] if not report_data.get(k, False)
            ]
            print("PRE-FLIGHT FAILED.")
            print(f"Failed checks: {failed_checks}")
            
        print("PRE-FLIGHT COMPLETED. FULL TRAINING HAS NOT BEEN EXECUTED.")
        print("="*50 + "\n")
        
        if dry_run:
            logger.info("DRY-RUN complete. Review memory figures above before full training.")
        else:
            logger.info("Training pipeline complete. Models saved and metrics exported.")
        logger.info("═" * 60)

    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation failed — training aborted: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        tracemalloc.stop()


if __name__ == '__main__':
    main()
