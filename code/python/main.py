import tools
import cv2
import numpy as np
import time
import sys
import json
import os

############################################################
# notes
# numpy.shape = (height, width, depth)
############################################################


img_dir = "../img/"
depth = 0


class Config(object):
    input_dir = "../../resource/image"
    output_dir = "../../output"
    texture_dir = "../../resource/image/texture"
    is_edge_darken = True
    is_wobble = True
    is_dry_brush = False


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
    # timer = Timer()
    cv2.pyrMeanShiftFiltering(input, 10, 25, input, maxLevel=1)
    # print(f"\tMeanshift using {timer.Tick()}")


def PerformAbstract(input):
    # timer = Timer()
    PerformMorph(input)
    PerformMeanshift(input)
    # print(f"\tAbstraction using {timer.Tick()}")


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
    # print(texture3D.shape)
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
    # print(texture3D.shape)
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
            if config.is_dry_brush and offset == 99:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
            else:
                offset = int(
                    0.03
                    * (
                        int(
                            texture[(i + texture_height) % texture_height][
                                (j + texture_width) % texture_width
                            ]
                        )
                        - 128
                    )
                )
                img[i][j] = img[i][(j + offset + 720) % 720]
    # cv2.imshow("wobble", img)
    # cv2.waitKey()
    return img


def Watercolorization(img, config: Config):
    timer = Timer()
    PerformAbstract(img)
    print(f"\tAbstraction \tusing \t{timer.Tick():.3f} sec")
    edge = cv2.Canny(img, 150, 200)
    print(f"\tEdgeDetect \tusing \t{timer.Tick():.3f} sec")
    # texture must have same shape
    if config.is_edge_darken:
        EdgeDarken(img, edge, os.path.join(config.texture_dir, "TextureCombined.jpg"))
    else:
        Render(img, config.texture_path)
    print(f"\tRendering \tusing \t{timer.Tick():.3f} sec")
    if config.is_wobble:
        wobble_texture = tools.LoadImage(
            "../../resource/image/texture/perlin_36.jpg", cv2.IMREAD_GRAYSCALE
        )
        img = Wobble(img, edge, wobble_texture, config)
    print(f"\tWobbling \tusing \t{timer.Tick():.3f} sec")
    return img


if __name__ == "__main__":
    ############################################################
    # Settings
    ############################################################
    config = Config()
    input_name = "01.png"
    input_path = "{}input/{}".format(config.input_dir, input_name)
    input_dir = config.input_dir
    ############################################################
    # Main Module
    ############################################################
    if len(sys.argv) == 2:
        # default config
        input_path = sys.argv[1]
        print(input_path)
    elif len(sys.argv) == 3:
        # use config
        json_config = json.load(sys.argv[1])
        input_path = sys.argv[2]
        config.input_dir = json_config["input_dir"]
        config.output_dir = json_config["output_dir"]
        config.texture_dir = json_config["texture_dir"]
        config.is_edge_darken = json_config["is_edge_darken"]
        config.is_dry_brush = json_config["is_dry_brush"]

    else:
        print("Wrong input format!\nYou can try commands below")

    if len(input_path.split("/")[-1].split(".")) < 2:
        # render all files in dir
        input_dir = input_path
        for (root, dirs, files) in os.walk(input_dir):
            dir = root[len(input_dir) + 1 :]
            for file in files:
                path = os.path.join(root, file)
                print(f"Rendering {path}")
                img = tools.LoadImage(path)
                img = Watercolorization(img, config)
                out_path = os.path.join(os.path.join(config.output_dir, dir), file)
                cv2.imwrite(out_path, img)
                print("finish\n")

    else:
        input_name = input_path.split("/")[-1]
        print(f"Rendering {input_path}")
        img = tools.LoadImage(input_path)
        img = Watercolorization(img, config)
        out_path = os.path.join(config.output_dir, input_name)
        cv2.imwrite(out_path, img)

    # img = tools.LoadImage(path)
    # img = Watercolorization(img, config)

    # cv2.imwrite(
    #     "{}Sample_{}_{}.png".format(config.output_dir, name, "origin"),
    #     img,
    # )
