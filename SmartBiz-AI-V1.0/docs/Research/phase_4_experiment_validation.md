# Phase 4 Experiment Validation Report

## SmartBiz AI V1.0

## Machine Learning Forecasting Model Evaluation


---

# 1. Research Objective

Phase 4 of SmartBiz AI V1.0 focuses on implementing and validating machine learning-based forecasting models for culinary demand prediction.

The main objective of this phase is to evaluate the capability of advanced ensemble machine learning algorithms in predicting future demand based on historical business data and supporting contextual features.

Three machine learning models were implemented and compared:

1. Random Forest Regressor
2. XGBoost Regressor
3. LightGBM Regressor

The evaluation process measures model performance using standardized forecasting metrics to identify the effectiveness of each model in estimating demand patterns.


---

# 2. Experimental Dataset

## 2.1 Dataset Description

The forecasting system uses structured daily demand data representing culinary business transactions.

The dataset contains historical information related to demand behavior and supporting business factors.

Main variables used in the forecasting pipeline include:

| Feature | Description |
|---|---|
| date | Transaction date information |
| sales_qty | Actual product demand quantity |
| price | Product selling price |
| promotion | Promotion activity indicator |
| holiday | Holiday information |
| weather | Weather condition information |
| stock_available | Available inventory quantity |


## 2.2 Dataset Preparation

Before model training, the dataset passed through preprocessing stages including:

- Data cleaning
- Missing value handling
- Feature transformation
- Feature preparation
- Separation between input features and target variable


The target variable for forecasting is:

which represents the predicted demand quantity.


---

# 3. Experimental Configuration

## 3.1 Data Splitting Strategy

The dataset was divided into training and testing subsets.

Configuration:

| Parameter | Value |
|-|-|
| Training Data | 80% |
| Testing Data | 20% |
| Validation Approach | Train-Test Split |


The training dataset was used for model learning, while the testing dataset was used to evaluate model generalization performance.


---

# 4. Machine Learning Models

## 4.1 Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions.

The model reduces variance and improves prediction stability by aggregating results from several independent trees.


## 4.2 XGBoost Regressor

XGBoost applies gradient boosting techniques by sequentially improving previous model errors.

This approach allows the model to capture complex nonlinear relationships between input features and demand patterns.


## 4.3 LightGBM Regressor

LightGBM is a gradient boosting framework designed for efficient learning with strong predictive performance.

The model uses tree-based learning optimization to capture complex feature interactions while maintaining computational efficiency.


---

# 5. Evaluation Metrics

The implemented models were evaluated using four forecasting metrics:

## Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted values and actual demand values.

Lower MAE indicates smaller prediction errors.


## Root Mean Square Error (RMSE)

RMSE measures prediction error magnitude while giving higher penalties to larger errors.


## Weighted Absolute Percentage Error (WAPE)

WAPE measures prediction accuracy relative to total actual demand.


## Symmetric Mean Absolute Percentage Error (sMAPE)

sMAPE evaluates relative forecasting error while reducing sensitivity toward scale differences.


---

# 6. Experimental Results

The final Phase 4 evaluation produced the following results:


| Model | MAE | RMSE | WAPE | sMAPE |
|-|-|-|-|-|
| Random Forest | 5.0875 | 5.9567 | 0.2609 | 26.1447 |
| XGBoost | 5.3233 | 6.3497 | 0.2730 | 27.0689 |
| LightGBM | 4.8833 | 5.8544 | 0.2504 | 25.0684 |


---

# 7. Result Analysis

The experimental results show that all implemented machine learning models successfully completed the forecasting pipeline.

Based on the evaluation metrics, LightGBM produced the lowest error values among the tested models.

LightGBM achieved:

- MAE: 4.8833
- RMSE: 5.8544
- WAPE: 0.2504
- sMAPE: 25.0684


These results indicate that LightGBM generated predictions with smaller deviation compared with Random Forest and XGBoost under the evaluated dataset configuration.

The performance difference suggests that gradient boosting-based approaches can effectively capture nonlinear relationships between historical demand patterns and supporting business features.


However, model selection will not rely only on numerical accuracy. Further analysis through feature importance evaluation and ablation study will be conducted to understand feature contribution and improve model interpretability.


---

# 8. Feature Importance Analysis

Feature importance analysis is conducted to identify the variables that contribute most significantly to demand prediction.

This analysis supports explainable AI implementation by providing insights into why the model generates specific predictions.


The feature importance results will be used to evaluate:

- Historical demand contribution
- Calendar influence
- Promotion impact
- Price influence
- Environmental factors
- Inventory factors


The output will support the development of SmartBiz AI decision recommendations.


---

# 9. Ablation Study Plan

The ablation study will evaluate the contribution of different feature groups toward forecasting performance.

The experimental scenarios are:


## Experiment A: Historical Demand Features

Input:

- Previous sales values
- Demand history patterns


Purpose:

Evaluate baseline forecasting capability using historical information only.


---

## Experiment B: Historical + Calendar Features

Additional features:

- Day of week
- Weekend indicator
- Holiday information


Purpose:

Measure the contribution of temporal patterns.


---

## Experiment C: Historical + Calendar + Business Features

Additional features:

- Promotion
- Price


Purpose:

Analyze the effect of business decision factors.


---

## Experiment D: Historical + Calendar + Business + Weather

Additional features:

- Weather information


Purpose:

Evaluate external environmental influence.


---

## Experiment E: Complete Feature Configuration

Additional features:

- Inventory availability
- Stock condition


Purpose:

Evaluate the complete SmartBiz AI forecasting scenario.


---

# 10. Phase 4 Validation Conclusion

Phase 4 successfully implemented and validated machine learning forecasting models for SmartBiz AI V1.0.

The complete pipeline has achieved:

- Successful model training
- Successful prediction generation
- Successful multi-model evaluation
- Successful experiment validation

The results provide a foundation for the next development stage, namely Phase 5 Decision Engine implementation.

Phase 5 will integrate forecasting outputs with inventory information and business constraints to generate actionable recommendations for culinary MSME decision support.