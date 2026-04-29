"""
EXP 2 – Decision Tree (Wine Dataset – Binary Classification)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Remove class 2 for binary classification
df = df[df['target'] != 2]

X = df.drop('target', axis=1)
y = df['target']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)

print("Accuracy:", accuracy_score(y_te, pred))

cm = confusion_matrix(y_te, pred)

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=wine.target_names[:2],
            yticklabels=wine.target_names[:2],
            cmap='PuBuGn')

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
