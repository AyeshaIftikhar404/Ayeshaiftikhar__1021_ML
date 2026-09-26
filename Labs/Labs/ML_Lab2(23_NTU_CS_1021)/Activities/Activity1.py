# ==========================================
# ACTIVITY 1
# Medical Insurance Cost Prediction
# Linear Regression
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load Dataset
data = pd.read_csv("Medical Cost Personal Datasets.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())


# ------------------------------------------
# Encode categorical variables
# ------------------------------------------

le_smoker = LabelEncoder()
le_region = LabelEncoder()

data["smoker"] = le_smoker.fit_transform(data["smoker"])
data["region"] = le_region.fit_transform(data["region"])


# ------------------------------------------
# Features and Target
# ------------------------------------------

X = data[["age", "bmi", "children", "smoker", "region"]]
y = data["charges"]


# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------
# Scale Features
# ------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------
# Train Linear Regression Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ------------------------------------------
# Prediction
# ------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------
# Evaluation
# ------------------------------------------

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRMSE:", rmse)
print("R2 Score:", r2)


# ------------------------------------------
# Plot Actual vs Predicted
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Insurance Cost")
plt.ylabel("Predicted Insurance Cost")
plt.title("Actual vs Predicted Insurance Costs")

plt.show()