# Phase 3 AI Research Pipeline

## 1. Objective
The objective of Phase 3 is to build a complete machine learning research pipeline for culinary MSME daily demand forecasting. This phase establishes the foundation for data preprocessing, feature engineering, baseline modeling, and evaluation metrics to support robust and reliable forecasting.

## 2. Architecture Overview
The Phase 3 AI Research Pipeline is structured into three main modules:
- **Preprocessing Pipeline** (`src/preprocessing/`): Handles data cleaning, transformation, and feature engineering.
- **Forecasting Module** (`src/forecasting/`): Implements baseline forecasting models and evaluation metrics.
- **Experiment Notebooks** (`notebooks/`): Provides interactive environments for exploratory data analysis, pipeline testing, and baseline model experimentation.
- **Result Storage** (`results/`): Directories for storing metrics, plots, and predictions.

## 3. Data Processing Pipeline
The data processing pipeline orchestrates the flow of data from raw CSV files to a clean, feature-rich dataset ready for model training. It consists of three sequential steps:
1. Data Cleaning
2. Feature Engineering
3. Train/Test Splitting

## 4. Data Cleaning Workflow
The data cleaning workflow (`src/preprocessing/cleaning.py`) ensures data quality by performing the following actions:
- Removing duplicate rows based on date and product identifiers.
- Converting date strings into proper datetime objects and sorting chronologically.
- Handling missing values using a specified strategy (e.g., linear interpolation).
- Detecting outliers using IQR or Z-score methods.

## 5. Feature Engineering
The feature engineering module (`src/preprocessing/feature_engineering.py`) generates relevant predictors for the forecasting models:
- **Calendar features:** Extracts day of the week (0-6), month (1-12), and a weekend flag (0/1) to capture weekly and monthly seasonality.
- **Promotion feature:** Ensures a binary promotion flag (0/1) is available to indicate whether a product was on promotion on a given day.
- **Lag features:** Creates lagged versions of the target variable (e.g., lag 1 and lag 7) to capture autoregressive dependencies.
- **Rolling features:** Computes rolling statistics (e.g., 7-day rolling mean) to smooth out daily fluctuations and highlight underlying trends.

## 6. Baseline Forecasting Models
Three baseline models are implemented in `src/forecasting/baseline.py` to establish a performance floor:
- **Seasonal Naive:** Predicts that future demand will exactly match the demand from the previous season (e.g., predicting next Monday's sales based on last Monday's sales).
- **Moving Average:** Forecasts future demand based on the average of the most recent `n` observations (e.g., a 7-day moving average).
- **Exponential Smoothing:** Applies simple exponential smoothing (SES) to generate a flat forecast, assigning exponentially decreasing weights to older observations.

## 7. Evaluation Metrics
The evaluation module (`src/forecasting/evaluation.py`) implements four standard metrics to assess forecast accuracy:
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual values.
- **RMSE (Root Mean Squared Error):** The square root of the average squared difference between predicted and actual values, penalizing larger errors more heavily.
- **WAPE (Weighted Absolute Percentage Error):** The sum of absolute errors divided by the sum of actual values, offering a robust alternative to MAPE.
- **sMAPE (Symmetric Mean Absolute Percentage Error):** A symmetric percentage error metric bounded between 0% and 200%, handling zero values more gracefully.

## 8. Experiment Notebook Structure
The `notebooks/` directory contains interactive Jupyter notebooks for research and experimentation:
- **`01_data_exploration.ipynb`:** Focuses on Exploratory Data Analysis (EDA), including sales trend visualization, distribution analysis, and temporal pattern analysis.
- **`02_preprocessing_pipeline.ipynb`:** Demonstrates the end-to-end preprocessing pipeline, including data cleaning, feature engineering, and train/test splitting.
- **`03_baseline_experiment.ipynb`:** Trains the baseline forecasting models, calculates evaluation metrics, visualizes predictions, and saves the results.

## 9. Testing Result
Phase 3 pipeline is fully operational with 16/16 tests passed.

## 10. Completion Checklist
- [x] Preprocessing pipeline created
- [x] Data cleaning and feature engineering implemented
- [x] Baseline forecasting models implemented
- [x] Evaluation metrics implemented
- [x] Experiment notebooks created
- [x] Result storage directories created
- [x] Comprehensive test suite executed and passed
- [x] Phase 3 documentation completed
