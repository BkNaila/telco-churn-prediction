# Telco Customer Churn Prediction

A classic Machine Learning project predicting customer churn for a telecom company, built to deepen my understanding of ML fundamentals.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, IBM) — 7,043 customers, ~73%/27% class imbalance (No/Yes churn).

## Approach

- **Cleaning**: fixed `TotalCharges` (missing values tied to `tenure = 0`), dropped `customerID`
- **Encoding**: binary mapping for 2-value columns, One-Hot Encoding for multi-category columns
- **Split**: train (70%) / validation (15%) / test (15%), stratified
- **Scaling**: `StandardScaler` on numerical features (fit on train only)
- **Models compared**: Logistic Regression, Random Forest, XGBoost, SVM — each with and without class balancing
- **Validation**: Stratified 5-Fold cross-validation on top models
- **Tuning**: `GridSearchCV` on the best model
- **Interpretability**: SHAP values to understand key churn drivers
- **Calibration**: checked reliability of predicted probabilities
- **Cost-sensitive threshold tuning**: optimized decision threshold based on business cost (missing a churner costs more than a false alarm)

## Best model

Logistic Regression (`C=0.1`, `penalty='l2'`, `class_weight='balanced'`)

| Metric | Validation (threshold 0.5) | Test (optimized threshold 0.25) |
|---|---|---|
| Accuracy | 0.741 | 0.622 |
| Precision | 0.507 | 0.408 |
| Recall | 0.814 | 0.936 |
| F1-score | 0.625 | 0.568 |

Key churn drivers (via SHAP): tenure, total charges, contract type.

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib/seaborn, Jupyter

## Next steps

- Model ensembling
- API deployment
