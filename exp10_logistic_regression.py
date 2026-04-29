"""
EXP 10 – Logistic Regression (Iris Dataset)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

X = df.drop('target', axis=1)
y = df['target']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)

lr = LogisticRegression(max_iter=200)
lr.fit(X_tr, y_tr)

pred = lr.predict(X_te)

print("Accuracy:", accuracy_score(y_te, pred))

cm = confusion_matrix(y_te, pred)

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cmap='coolwarm')

plt.title("Logistic Regression")
plt.tight_layout()
plt.show()
