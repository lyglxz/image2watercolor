import abstraction
import tools
import tester
import cv2
import numpy as np
import os
import time
import traceback

############################################################
# notes
# numpy.shape = (height, width, depth)
############################################################


img_dir = "../img/"
depth = 0


class Config(object):
    input_dir = "../../resource/image/"
    output_dir = "../../output/"
    texture_path = "../../resource/image/texture/TextureCombined.jpg"
    is_edge_darken = True
    is_wobble = True
    pass


class Timer(object):
    def __init__(self) -> None:
        self.tick = time.time()

    def Tick(self):
        ret = self.tick
        tick = time.time()
        return tick - ret


def PerformMorph(input):
    # timer = Timer.time()
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    progress = [cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_CLOSE]
    for op in progress:
        cv2.morphologyEx(input, op, element, input, iterations=2)
    # print(f"\tMorph using {timer.Tick()}")


def PerformMeanshift(input):
    timer = Timer()
    cv2.pyrMeanShiftFiltering(input, 10, 25, input, maxLevel=1)
    print(f"\tMeanshift using {timer.Tick()}")


def PerformAbstract(input):
    timer = Timer()
    PerformMorph(input)
    PerformMeanshift(input)
    print(f"\tAbstraction using {timer.Tick()}")


def EdgeDarken(img, edge, noise_path=None):
    if noise_path is None:
        texture = tools.LoadImage(
            "../img/pipeline/TextureCombined.jpg", cv2.IMREAD_GRAYSCALE
        )
    else:
        texture = tools.LoadImage(noise_path, cv2.IMREAD_GRAYSCALE)
    texture = texture * edge * 5.0 / (255.0) + texture
    # texture = edge / 10.0 + 128
    texture3D = np.stack((texture, texture, texture), axis=2)
    # img = np.float32(img)
    print(texture3D.shape)
    ret = img * (1 - (1 - img / 255) * (2 * texture3D / 255 - 1))
    return ret


def Render(img, noise_path=None):
    if noise_path is None:
        texture = tools.LoadImage(
            "../img/pipeline/TextureCombined.jpg", cv2.IMREAD_GRAYSCALE
        )
    else:
        texture = tools.LoadImage(noise_path, cv2.IMREAD_GRAYSCALE)
    texture3D = np.stack((texture, texture, texture), axis=2)
    img = np.float32(img)
    print(texture3D.shape)
    ret = img * (1 - (1 - img / 255) * (2 * texture3D / 255 - 1))
    return ret


def Wobble(img, edge, texture, config):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.morphologyEx(edge, cv2.MORPH_DILATE, element, edge, iterations=2)
    (img_height, img_width) = img.shape[0:2]
    (texture_height, texture_width) = texture.shape[0:2]
    for i in range(img_height):
        for j in range(img_width):
            if edge[i][j] != 255:
                continue
            offset = texture[i][j]
            if is_dry_brush and offset == 99:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
            else:
                offset = int(
                    0.03
                    * (
                        int(
                            edge[(i + texture_height) % texture_height][
                                (j + texture_width) % texture_width
                            ]
                        )
                        - 128
                    )
                )
                img[i][j] = img[i][(j + offset + 720) % 720]
    return img


def Watercolorization(img, config):
    PerformAbstract(img)
    edge = cv2.Canny(img, 150, 200)
    if config.is_edge_darken:
        EdgeDarken(img, edge, config.texture_path)
    else:
        Render(img, config.texture_path)

    if config.is_wobble:
        wobble_texture = tools.LoadImage(
            "../../resource/image/texture/perlin_36.jpg", cv2.IMREAD_GRAYSCALE
        )
        img = Wobble(img, edge, wobble_texture, config)
    return img


if __name__ == "__main__":
    ############################################################
    # Settings
    ############################################################
    is_dry_brush = False
    # Watercolorization(None, None)

    ############################################################
    # Main Module
    ############################################################
    # cv2.setUseOptimized(onoff=True)
    file_name = "b1.png"
    config = Config()
    [name, end] = file_name.split(".")
    path = "{}input/{}".format(config.input_dir, file_name)
    img = tools.LoadImage(path)
    img = Watercolorization(img, config)

    cv2.imwrite(
        "{}Sample_{}_{}.png".format(config.output_dir, name, "origin"),
        img,
    )
    # timer = Timer()

    # PerformAbstract(img)
    # print(f"Abstraction\n{timer.Tick()} seconds\n")
    # cv2.imwrite(
    #     "../img/output/Sample_{}_{}.png".format(name, "Abstraction"),
    #     img[450:700, 350:500],
    # )
    # edge = cv2.Canny(img, 150, 200)
    # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # cv2.imwrite("../img/output/{}_{}.png".format(name, "edge"), edge)
    # cv2.morphologyEx(edge, cv2.MORPH_DILATE, element, edge, iterations=2)
    # noise_perlin = tools.LoadImage(
    #     "../img/output/NoisePerlin/perlin_72.jpg", cv2.IMREAD_GRAYSCALE
    # )
    # cv2.imwrite("../img/output/{}_{}.png".format(name, "edgeDia"), edge)
    # cv2.imwrite(
    #     "../img/output/Sample_{}_{}.png".format(name, "edgeDia"),
    #     edge[450:700, 350:500],
    # )

    # # img = Render(img, "../img/pipeline/TextureCombined_old.jpg")
    # # print(f"Render\n{timer.Tick()} seconds\n\n")
    # # cv2.imwrite(
    # #     "../img/output/Sample_{}_{}.png".format(name, "Render"), img[450:800, 350:700]
    # # )

    # img = EdgeDarken(img, edge, "../img/pipeline/TextureCombined_old.jpg")
    # print(f"EdgeDarken\n{timer.Tick()} seconds\n\n")
    # cv2.imwrite(
    #     "../img/output/Sample_{}_{}.png".format(name, "EdgeDarken"),
    #     img[450:700, 350:500],
    # )
    # cv2.imwrite(
    #     "../img/output/{}_{}.png".format(name, "EdgeDarken_compare"),
    #     img[500:700, 550:750],
    # )
    # count = 0
    # for i in range(1080):
    #     for j in range(720):
    #         if edge[i][j] != 255:
    #             continue
    #         offset = noise_perlin[i][j]
    #         if is_dry_brush and offset == 99:
    #             img[i][j][0] = 255
    #             img[i][j][1] = 255
    #             img[i][j][2] = 255
    #         else:
    #             offset = int(0.03 * (int(noise_perlin[i][j]) - 128))
    #             img[i][j] = img[i][(j + offset + 720) % 720]
    #         count = count + 1
    # print(count)
    # cv2.imwrite("../img/output/{}_{}.png".format(name, "Humble_0.03"), img)
    # cv2.imwrite(
    #     "../img/output/{}_{}.png".format(name, "Cut_Humble_0.03"), img[0:550, 240:480]
    # )
    ############################################################
    # Pipe Line
    ############################################################
    # save_dir = "../img/demo/out"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # i = 0
    # for root, dirs, files in os.walk("../img/demo", topdown=False):
    #     if root.endswith("out"):
    #         continue
    #     for file_name in files:
    #         [name, end] = file_name.split(".")
    #         path = os.path.join(root, file_name)
    #         img = tools.LoadImage(path)
    #         timer = Timer()
    #         PerformAbstract(img)
    #         print(f"Abstraction\n{timer.Tick()} seconds\n")
    #         edge = cv2.Canny(img, 75, 175)
    #         element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #         # cv2.imwrite("../img/output/{}_{}.png".format(name, "edge"), edge)
    #         cv2.morphologyEx(edge, cv2.MORPH_DILATE, element, edge, iterations=2)
    #         # cv2.imwrite("../img/output/{}_{}.png".format(name, "edgeDia"), edge)

    #         # img = Render(img, "../img/pipeline/TextureCombined.jpg")
    #         img = EdgeDarken(img, edge, "../img/pipeline/TextureCombined_old.jpg")

    #         print(f"Render\n{timer.Tick()} seconds\n\n")
    #         # cv2.imshow("demo", img)
    #         # cv2.waitKey()
    #         cv2.imwrite("../img/demo/out/{}.png".format(file_name), img)

    ############################################################
    # Module Test
    ############################################################
    # tools.GetCombinedTexture(img_dir + "pipeline")
    # tools.GetDryBrush("../img/pipeline")

    ############################################################
    # Do Tests and generate results at ../img/output
    ############################################################
    # Tester.TestKMeans(path, "../img/output/KMeans")
    # Tester.TestGrid(path, "../img/output/Grid")
    # Tester.TestMeanShift(path, "../img/output/Meanshift")
    # Tester.TestMorph(path, "../img/output/Morph")
    # Tester.TestColorDensity("../img/output/Color")
    # Tester.TestNoiseGauss("../img/output/NoiseGauss")
    # Tester.TestNoisePerlin("../img/output/NoisePerlin")
