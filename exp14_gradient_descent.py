"""
EXP 14 – Gradient Descent (Linear Regression from Scratch)
"""

import numpy as np

X = np.array([32.5, 53.4, 61.5, 47.4, 59.8])
Y = np.array([31.7, 68.7, 62.5, 71.5, 87.2])

w, b = 0.0, 0.0
lr = 0.01

for i in range(1000):
    y_pred = w * X + b
    error = Y - y_pred

    w += lr * (X * error).mean()
    b += lr * error.mean()

print("Weight:", w)
print("Bias:", b)
