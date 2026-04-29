"""
EXP 3 – Random Forest (Digits Dataset)
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

X, y = load_digits(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25)

rf = RandomForestClassifier(random_state=23)
rf.fit(X_tr, y_tr)

y_pred = rf.predict(X_te)

print("Accuracy:", accuracy_score(y_te, y_pred))

cm = confusion_matrix(y_te, y_pred)

sns.heatmap(cm, annot=True, cmap='winter')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
