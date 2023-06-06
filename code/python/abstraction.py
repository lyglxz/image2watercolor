import cv2
import numpy as np


def KMeans(src, k, max_iter=10, eps=1.0):
    dst = src.reshape((-1, 3))
    dst = np.float32(dst)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    ret, label, center = cv2.kmeans(
        dst,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    center = np.uint8(center)
    dst = center[label.flatten()]
    dst = dst.reshape((src.shape))
    # cv2.imshow("res", dst)
    # cv2.waitKey()
    return dst
