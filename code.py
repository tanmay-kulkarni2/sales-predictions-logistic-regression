# ================================
# Sales Prediction using Logistic Regression
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
dataset = pd.read_csv('sales_dataset.csv') 
print("Dataset Shape:", dataset.shape)
print(dataset.head())

# Independent & Dependent Variables
X = dataset.iloc[:, :-1].values   # Age, Salary
y = dataset.iloc[:, -1].values    # Status (0/1)

# ================================
# Preprocessing
# ================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=55, stratify=y
)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ================================
# Logistic Regression with Hyperparameter Tuning
# ================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid = GridSearchCV(LogisticRegression(random_state=32), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# ================================
# Predictions
# ================================
y_pred = best_model.predict(X_test)

# ================================
# Evaluation
# ================================
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report, roc_curve
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ================================
# Coefficients as Odds Ratios

odds_ratios = np.exp(best_model.coef_[0])
for feature, coef, odds in zip(['Age','Salary'], best_model.coef_[0], odds_ratios):
    print(f"{feature}: Coefficient={coef:.4f}, Odds Ratio={odds:.4f}")

# ================================
# Example Prediction for New Customer
# ================================

age = 35
salary = 50000
new_customer = sc.transform([[age, salary]])
prediction = best_model.predict(new_customer)[0]
print(f"Prediction for Age={age}, Salary={salary}: {'Will Buy' if prediction==1 else 'Won’t Buy'}")
