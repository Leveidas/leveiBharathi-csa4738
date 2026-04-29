"""
EXP 7 – Sigmoid Function
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid_func(z):
    return 1 / (1 + np.exp(-z))

x_vals = np.arange(-5, 5, 0.1)
y_vals = sigmoid_func(x_vals)

plt.plot(x_vals, y_vals, color='pink')
plt.title("Sigmoid Curve")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid(True)
plt.tight_layout()
plt.show()
