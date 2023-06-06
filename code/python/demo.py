import numpy as np
import cv2
import sys


img = np.ones((1080, 720, 3)) * 255
img[:, 0:350] = (255, 0, 0)
img[:, 370:] = (0, 0, 255)

noise = cv2.imread("../img/output/NoisePerlin/perlin_36.jpg", cv2.IMREAD_GRAYSCALE)

for i in range(1080):
    for j in range(720):
        if img[i][j][1] != 255:
            continue
        offset = (int(noise[i][j]) - 128) // 3
        img[i][j] = img[i][j + offset]
cv2.imwrite("./demo_36_divide_10.png", img)
