# ==========================================
# ACTIVITY 3
# Used Car Price Prediction
# Linear + Polynomial Regression
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------
# Load Dataset
# ------------------------------------------

data = pd.read_csv("Car Price Prediction.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)


# ------------------------------------------
# Features and Target
# ------------------------------------------

X = data[
    [
        "enginesize",
        "horsepower",
        "curbweight",
        "citympg"
    ]
]

y = data["price"]


# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)


# Evaluation

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)


print("\n================================")
print("LINEAR REGRESSION")
print("================================")

print("RMSE:", linear_rmse)

print("R2 Score:", linear_r2)


# ==========================================
# POLYNOMIAL REGRESSION
# Degrees 2, 3 and 4
# ==========================================

results = []


for degree in [2, 3, 4]:

    print("\n================================")
    print("POLYNOMIAL REGRESSION")
    print("Degree:", degree)
    print("================================")


    # Polynomial Features

    poly = PolynomialFeatures(
        degree=degree
    )


    X_train_poly = poly.fit_transform(
        X_train
    )

    X_test_poly = poly.transform(
        X_test
    )


    # Train model

    poly_model = LinearRegression()

    poly_model.fit(
        X_train_poly,
        y_train
    )


    # Prediction

    poly_pred = poly_model.predict(
        X_test_poly
    )


    # Evaluation

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            poly_pred
        )
    )

    r2 = r2_score(
        y_test,
        poly_pred
    )


    print("RMSE:", rmse)

    print("R2 Score:", r2)


    # Store results

    results.append(
        [
            degree,
            rmse,
            r2
        ]
    )


# ==========================================
# Comparison Table
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Polynomial Degree",
        "RMSE",
        "R2 Score"
    ]
)


print("\n================================")
print("POLYNOMIAL COMPARISON")
print("================================")

print(results_df)

# ==========================================
# Plot Actual vs Predicted
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    linear_pred
)

plt.xlabel("Actual Car Price")

plt.ylabel("Predicted Car Price")

plt.title(
    "Actual vs Predicted Car Prices"
)

plt.show()



# ==========================================
# FITTED CURVES FOR POLYNOMIAL DEGREES
# ==========================================

# We will use horsepower for visualization
X_curve = data[["horsepower"]]
y_curve = data["price"]

# Sort values for smooth curves
sort_index = X_curve["horsepower"].argsort()

X_sorted = X_curve.iloc[sort_index]
y_sorted = y_curve.iloc[sort_index]

plt.figure(figsize=(10, 6))

# Plot actual data
plt.scatter(
    X_sorted["horsepower"],
    y_sorted,
    alpha=0.5,
    label="Actual Data"
)

# Plot polynomial curves
for degree in [2, 3, 4]:

    poly = PolynomialFeatures(
        degree=degree
    )

    X_poly = poly.fit_transform(
        X_sorted
    )

    model = LinearRegression()

    model.fit(
        X_poly,
        y_sorted
    )

    y_curve_pred = model.predict(
        X_poly
    )

    plt.plot(
        X_sorted["horsepower"],
        y_curve_pred,
        label=f"Degree {degree}"
    )

# Labels
plt.xlabel("Horsepower")
plt.ylabel("Car Price")

plt.title(
    "Polynomial Regression Fitted Curves"
)

plt.legend()

plt.show()