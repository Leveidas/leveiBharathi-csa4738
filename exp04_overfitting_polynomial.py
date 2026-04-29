"""
EXP 4 – Overfitting (Polynomial Regression)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(0)

def func(x):
    return np.cos(1.5 * np.pi * x)

x = np.sort(np.random.rand(30))
y = func(x) + np.random.randn(30) * 0.1

degrees = [1, 4, 15]

plt.figure(figsize=(12, 4))

for i, d in enumerate(degrees):
    plt.subplot(1, 3, i + 1)

    pipe = Pipeline([
        ("poly", PolynomialFeatures(d)),
        ("lr", LinearRegression())
    ])

    pipe.fit(x.reshape(-1, 1), y)

    x_test = np.linspace(0, 1, 100)
    y_pred = pipe.predict(x_test.reshape(-1, 1))

    plt.plot(x_test, y_pred)
    plt.scatter(x, y)
    plt.title(f"Degree {d}")

plt.tight_layout()
plt.show()
