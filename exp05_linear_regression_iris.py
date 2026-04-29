"""
EXP 5 – Linear Regression (Iris Dataset)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

X = df[['sepal length (cm)']]
y = df['sepal width (cm)']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(X_tr, y_tr)

y_pred = lr.predict(X_te)

print("MSE:", mean_squared_error(y_te, y_pred))

plt.scatter(X_te, y_te)
plt.plot(X_te, y_pred, color='red')
plt.title("Linear Regression")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()
