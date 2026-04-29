"""
EXP 1 – Confusion Matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

true_vals = np.array(['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
pred_vals = np.array(['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])

cm = confusion_matrix(true_vals, pred_vals)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g',
            xticklabels=['Dog', 'Not Dog'],
            yticklabels=['Dog', 'Not Dog'],
            cmap='RdPu')

plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
