"""
EXP 15 – Image Segmentation using KMeans (OpenCV)

Note: Place an image named 'image.jpg' in the same directory before running.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = rgb.reshape((-1, 3))
pixels = np.float32(pixels)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3

_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented = centers[labels.flatten()]
segmented = segmented.reshape(rgb.shape)

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title("Segmented")

plt.tight_layout()
plt.show()
