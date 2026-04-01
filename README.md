# Sales-Predictions-Logistic-Regression
A machine learning project that applies Logistic Regression to predict sales outcomes based on features such as customer demographics, and product categories. Includes end-to-end workflow: data preprocessing, model training, evaluation, and visualization. Demonstrates how statistical modeling can support business forecasting and decision-making.

# Sales Prediction using Logistic Regression

## Overview
This project applies Logistic Regression, a fundamental machine learning algorithm, to predict whether sales will occur based on given features. It demonstrates how statistical modeling can support business decision-making in sales forecasting.

## Objectives
- Predict sales outcomes (Yes/No) using logistic regression.
- Understand the relationship between independent variables (e.g., marketing spend, customer demographics, product category) and sales.
- Provide a reproducible workflow for business-focused machine learning projects.

## Dataset
- **Source**: "DigitalAd_dataset"
- **Features**: Examples include marketing spend, region, product type, customer age group, etc.
- **Target Variable**: Binary outcome (Sale = 1, No Sale = 0).

## Methodology
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
2. **Model Building**
   - Train-test split
   - Logistic Regression implementation
   - Model training and evaluation
3. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC Curve

## Results
- Model performance metrics
- Insights into which features most influence sales
- Visualization of predictions and decision boundaries

## Requirements
- **Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Features Used
- Feature Scaling → ensures Age/Salary are on comparable scales.
- Hyperparameter Tuning (GridSearchCV) → finds best C, penalty, solver.
- Cross-validation → more robust evaluation.
- Advanced Metrics → precision, recall, F1, ROC-AUC.
- Visualizations → confusion matrix heatmap, ROC curve.
- Business Insights → odds ratios show how Age/Salary affect buying likelihood.
- Stratified Split → keeps class balance in train/test.

## Why Stratified Split Matter ?
  - If the dataset has 70% "0" (No Sale) and 30% "1" (Sale)
  - Without stratification, the random split might accidentally produce a test set with 90% "0" and only 10% "1". That            would make your evaluation misleading because the test set doesn’t represent the true distribution.
  - With stratify=y, both train and test sets will maintain approximately the same 70/30 ratio.

## Parameters 
  - C (Inverse of Regularization Strength)
          - Controls how much penalty is applied to the model coefficients.
          - Smaller values (0.01, 0.1) → stronger regularization (simpler model, less overfitting).
          - Larger values (1, 10) → weaker regularization (model fits data more closely, risk of overfitting).
- penalty (Type of Regularization)
          - 'l1' → Lasso regularization: can shrink some coefficients to zero, effectively performing feature selection.
          - 'l2' → Ridge regularization: shrinks coefficients but keeps all features, useful for multicollinearity.
- solver (Optimization Algorithm)
          - 'liblinear' → A solver that supports both L1 and L2 penalties, good for small to medium dataset.
          - Different solvers are optimized for different dataset sizes and penalty types, but here liblinear is the right                 choice.

 ##Importance.
    - GridSearchCV will try every combination of these parameters (C × penalty × solver).
    - It will train multiple models and pick the one with the best performance based on cross-validation.
    - This ensures your logistic regression is optimized, not just using default settings.

 


