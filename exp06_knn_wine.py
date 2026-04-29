"""
EXP 6 – K-Nearest Neighbors (Wine Dataset)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['label'] = wine.target

X = df.drop('label', axis=1)
y = df['label']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tr, y_tr)

pred = knn.predict(X_te)

print("Accuracy:", accuracy_score(y_te, pred))

cm = confusion_matrix(y_te, pred)

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names,
            cmap='summer')

plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.show()
