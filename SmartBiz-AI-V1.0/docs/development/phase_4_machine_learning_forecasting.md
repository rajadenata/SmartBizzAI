# Phase 4 Machine Learning Forecasting Models

## 1. Objective
The objective of Phase 4 is to implement advanced machine learning algorithms (Random Forest, XGBoost, and LightGBM) to improve demand forecasting accuracy over the baseline models established in Phase 3. This phase introduces an automated training pipeline and feature importance analysis while maintaining compatibility with the existing evaluation metrics and project architecture.

## 2. Architecture Overview
The Phase 4 implementation adds machine learning capabilities without disrupting existing modules:
- **ML Models Module** (`src/forecasting/ml_models.py`): Contains wrapper classes for Random Forest, XGBoost, and LightGBM regressors with a unified interface (`fit`, `predict`, `get_feature_importance`).
- **Training Pipeline** (`src/forecasting/training_pipeline.py`): Orchestrates data preparation, model training, prediction generation, and results saving.
- **Experiment Notebooks** (`notebooks/Training/04_ml_model_experiment.ipynb`): Provides an interactive environment for comparing ML models and analyzing feature importances.
- **Result Storage** (`results/models/`, `results/feature_importance/`): New directories for saving serialized models and feature importance CSVs.

## 3. Machine Learning Models
Three powerful tree-based ensemble models were implemented:
- **Random Forest Regressor:** A robust bagging algorithm that averages multiple decision trees to reduce variance and prevent overfitting.
- **XGBoost Regressor:** An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.
- **LightGBM Regressor:** A fast, distributed, high-performance gradient boosting framework that uses tree-based learning algorithms, particularly effective on larger datasets.

All models adhere to the `MLModelWrapper` base class, ensuring a consistent interface.

## 4. Training Pipeline
The training pipeline automates the end-to-end model execution:
1. **Feature Preparation:** Extracts relevant features and separates the target variable, automatically handling missing values (e.g., from initial lag shifts) and dropping non-feature columns (`date`, `product_id`).
2. **Model Training:** Fits all three ML models on the training dataset.
3. **Prediction:** Generates demand forecasts on the chronological test set, enforcing non-negativity.
4. **Feature Importance:** Extracts feature importance scores from the tree-based models to interpret which factors (e.g., lag, promotion, day of week) drive demand.

## 5. Experiment Notebook
The new `04_ml_model_experiment.ipynb` notebook demonstrates:
- End-to-end execution of the pipeline using preprocessed Phase 3 data.
- Evaluation of ML models using existing Phase 3 metrics (MAE, RMSE, WAPE, sMAPE).
- Visual comparison of predictions against actual sales and baseline models.
- Visualization of feature importances, showing the impact of lag features and promotions.

## 6. Testing Result
The Phase 4 pipeline is fully operational with all unit tests passing. The test suite (`tests/test_phase4_pipeline.py`) covers data preparation, individual model initialization, fitting, prediction, feature importance extraction, and pipeline orchestration.

## 7. Completion Checklist
- [x] Implemented Random Forest, XGBoost, and LightGBM models
- [x] Created `MLModelWrapper` for a unified interface
- [x] Integrated models with Phase 3 preprocessing
- [x] Created `training_pipeline.py`
- [x] Implemented feature importance extraction
- [x] Set up result storage directories (`results/models/`, `results/feature_importance/`)
- [x] Created ML experiment notebook (`04_ml_model_experiment.ipynb`)
- [x] Added Phase 4 unit tests
- [x] Wrote Phase 4 development documentation
