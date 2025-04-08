import cv2
import numpy as np

label = np.load("models/label.npy")

img = cv2.imread('Dataset/SatelliteImages/3.png')
mask = label[3]
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
print(mask.shape)
masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("aa",cv2.resize(masked,(250,250)))
cv2.waitKey(0)
