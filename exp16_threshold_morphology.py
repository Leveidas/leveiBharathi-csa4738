"""
EXP 16 – Thresholding + Morphological Operations (OpenCV)

Note: Place an image named 'image.jpg' in the same directory before running.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((2, 2), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

plt.imshow(closing, cmap='gray')
plt.title("Threshold + Morphology")
plt.tight_layout()
plt.show()
