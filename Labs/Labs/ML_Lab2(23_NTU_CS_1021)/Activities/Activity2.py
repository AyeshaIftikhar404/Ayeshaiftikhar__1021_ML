# ==========================================
# ACTIVITY 2
# Customer Churn Prediction
# Logistic Regression
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# ------------------------------------------
# Load Dataset
# ------------------------------------------

data = pd.read_csv("Telco Customer Churn.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)


# ------------------------------------------
# Convert TotalCharges to numeric
# ------------------------------------------

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove missing values
data = data.dropna()


# ------------------------------------------
# Convert Churn to 0 and 1
# ------------------------------------------

data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ------------------------------------------
# Select Features
# ------------------------------------------

X = data[
    [
        "tenure",
        "MonthlyCharges",
        "Contract",
        "InternetService"
    ]
]

y = data["Churn"]


# ------------------------------------------
# Encode categorical variables
# ------------------------------------------

X = pd.get_dummies(
    X,
    columns=["Contract", "InternetService"],
    drop_first=True
)

# Convert True/False to 1/0
X = X.astype(int)


# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------
# Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------
# Logistic Regression
# ------------------------------------------

model = LogisticRegression()

model.fit(X_train, y_train)


# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------
# Evaluation
# ------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n===== Evaluation Results =====")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)


# ------------------------------------------
# Confusion Matrix
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.show()


# ------------------------------------------
# ROC Curve
# ------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(
    fpr,
    tpr
)


plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label="ROC Curve (AUC = {:.2f})".format(roc_auc)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


print("AUC:", roc_auc)