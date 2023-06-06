import cv2
import numpy as np
import time
from perlin_numpy import generate_perlin_noise_2d
import os


def LoadImage(path: str, color=cv2.IMREAD_COLOR, size=(720, 1080)):
    img = cv2.imread(path, color)
    shape = img.shape
    # print(img.shape)
    # need to reshape
    img = cv2.resize(img, (int(size[1] * shape[1] / shape[0]), size[1]))

    if img.shape[1] < size[0]:
        print("pad")
        img = cv2.copyMakeBorder(
            img, 0, size[1] - img.shape[1], 0, 0, cv2.BORDER_DEFAULT
        )
    elif img.shape[1] > size[0]:
        start = int(img.shape[1] / 2 - size[0] / 2)
        end = int(img.shape[1] / 2 + size[0] / 2)
        img = img[:, start:end]
    return img


def ColorGrad(c: float, density, op=0):
    # print(c, density)
    if op == 0:
        return c * (1 - (1 - c) * (density - 1))
    else:
        SUQARE_MAX = 65025
        return (
            c
            * (SUQARE_MAX * 2 - 510 * density + 2 * density * c - 255 * c)
            / SUQARE_MAX
        )


def GetPerlinNoise(size=(1080, 720), res=(20, 20), tileable=(False, True)):
    np.random.seed(int(time.time()))
    noise = generate_perlin_noise_2d(size, res)
    # print(np.min(noise))
    for e in np.nditer(noise, op_flags=["readwrite"]):
        e[...] = (e + 1) / 2
    return noise


def GetCombinedTexture(
    src_dir,
    src_files=["gauss.jpg", "perlin.jpg", "paper.jpg"],
    percent=[25, 5, 70],
    size=(720, 1080),
):
    sum = 0.0
    for e in percent:
        sum = sum + e
    dst = np.zeros((size[1], size[0]))
    for i in range(len(src_files)):
        tmp = cv2.imread(os.path.join(src_dir, src_files[i]), cv2.IMREAD_GRAYSCALE)
        dst = dst + percent[i] / sum * tmp
    cv2.imwrite(os.path.join(src_dir, "TextureCombined_old.jpg"), dst)


def GetDryBrush(src_dir, src_file="paper.jpg", thres=200):
    img = LoadImage(os.path.join(src_dir, src_file), cv2.IMREAD_GRAYSCALE)
    cv2.threshold(img, 190, 255, type=cv2.THRESH_BINARY, dst=img)
    cv2.imshow("out", img)
    cv2.waitKey()
    pass
