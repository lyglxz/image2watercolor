import abstraction
import tools
import cv2
import os
import numpy as np


def TestKMeans(input_file, save_dir, test_list=[2, 6, 12, 24, 48]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    [file_name, end] = os.path.basename(input_file).split(".")
    img = tools.LoadImage(input_file)
    save_path = "{}/{}_{}.{}".format(save_dir, file_name, "origin", end)
    cv2.imwrite(save_path, img)
    for i in test_list:
        ret = Abstraction.KMeans(img, i)
        save_path = "{}/{}_{}.{}".format(save_dir, file_name, i, end)
        cv2.imwrite(save_path, ret)


def TestGrid(
    input_file,
    save_dir,
    test_list=[2, 8, 16, 32, 64],
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    [file_name, end] = os.path.basename(input_file).split(".")
    img = tools.LoadImage(input_file)
    save_path = "{}/{}_{}.{}".format(save_dir, file_name, "origin", end)
    cv2.imwrite(save_path, img)
    for k in test_list:
        ret = img
        for e in np.nditer(ret, op_flags=["readwrite"]):
            tail = e % k
            e[...] = e - tail + int(tail / k)
        save_path = "{}/{}_{}.{}".format(save_dir, file_name, k, end)
        cv2.imwrite(save_path, ret)


def TestMeanShift(input_file, save_dir, sr=[10, 50], sp=[3, 10], max_level=[0, 1, 2]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    [file_name, end] = os.path.basename(input_file).split(".")
    img = tools.LoadImage(input_file)
    save_path = "{}/{}_{}.{}".format(save_dir, file_name, "origin", end)
    cv2.imwrite(save_path, img)
    for j in sr:
        for k in sp:
            for i in max_level:
                ret = img
                cv2.pyrMeanShiftFiltering(img, k, j, ret, maxLevel=i)
                save_path = "{}/{}_sp_{}_sr_{}_maxlevel_{}.{}".format(
                    save_dir, file_name, k, j, i, end
                )
                cv2.imwrite(save_path, ret)


def TestMorph(
    input_file,
    save_dir,
    test_list=[
        "",
    ],
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    [file_name, end] = os.path.basename(input_file).split(".")
    img = tools.LoadImage(input_file)
    save_path = "{}/{}_{}.{}".format(save_dir, file_name, "origin", end)
    cv2.imwrite(save_path, img)

    # preset
    ret = img.copy()
    tail = ""
    save_ret = f"{save_dir}/{file_name}_" + "{}" + f".{end}"

    # generate elements
    element3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    element5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    element7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    element_dic = {3: element3, 5: element5, 7: element7}

    print(element7, "\n\n")
    print(element3)

    cv2.morphologyEx(ret, cv2.MORPH_CLOSE, element_dic[3], ret, iterations=3)
    cv2.morphologyEx(ret, cv2.MORPH_OPEN, element_dic[3], ret, iterations=3)
    cv2.imwrite(save_ret.format("C3C3C3O3O3O3_slice"), ret[540:810, 360:540])
    ret = img.copy()

    cv2.morphologyEx(ret, cv2.MORPH_OPEN, element_dic[3], ret, iterations=3)
    cv2.morphologyEx(ret, cv2.MORPH_CLOSE, element_dic[3], ret, iterations=3)
    cv2.imwrite(save_ret.format("O3O3O3C3C3C3_slice"), ret[540:810, 360:540])
    ret = img.copy()

    for i in range(3):
        cv2.morphologyEx(ret, cv2.MORPH_CLOSE, element_dic[3], ret, iterations=1)
        cv2.morphologyEx(ret, cv2.MORPH_OPEN, element_dic[3], ret, iterations=1)
    cv2.imwrite(save_ret.format("C3O3C3O3C3O3"), ret)
    ret = img.copy()

    # cv2.morphologyEx(ret, cv2.MORPH_CLOSE, element_dic[7], ret, iterations=1)
    # cv2.imwrite(save_ret.format("C7_slice"), ret[540:810, 360:540])
    # ret = img.copy()

    # morph_op = cv2.MORPH_OPEN
    # for k in element_dic:
    #     cv2.morphologyEx(ret, morph_op, element_dic[k], ret, iterations=1)
    #     tail = "O" + str(k)
    #     cv2.imwrite(save_ret.format(tail), ret)
    #     ret = img.copy()

    # morph_op = cv2.MORPH_CLOSE
    # for k in element_dic:
    #     cv2.morphologyEx(ret, morph_op, element_dic[k], ret, iterations=1)
    #     tail = "C" + str(k)
    #     cv2.imwrite(save_ret.format(tail), ret)
    #     ret = img.copy()

    # cv2.morphologyEx(ret, cv2.MORPH_OPEN, element7, ret, iterations=1)
    # tail = "O7"
    # cv2.imwrite(save_ret.format(tail), ret)
    # ret = img.copy()
    # cv2.morphologyEx(ret, cv2.MORPH_OPEN, element3, ret, iterations=1)
    # tail = "O3"
    # cv2.imwrite(save_ret.format(tail), ret)
    # ret = img.copy()
    # cv2.morphologyEx(ret, cv2.MORPH_OPEN, element5, ret, iterations=1)
    # tail = "O5"
    # cv2.imwrite(save_ret.format(tail), ret)
    # ret = img.copy()
    # cv2.imshow("img", img)
    # cv2.waitKey()

    # cv2.morphologyEx(img, cv2.MORPH_OPEN, morph_element2, ret, iterations=iter2)
    # cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_element1, ret, iterations=iter1)
    # save_path = "{}/{}_{}_{}_({}_{}).{}".format(
    #     save_dir, file_name, "CO", iter2, size1[0], size1[1], end
    # )
    # save_path = "{}/{}_{}_{}.{}".format(save_dir, file_name, "CO", 4, end)
    # cv2.imwrite(save_path, ret)


def TestColorDensity(save_dir, demo_color=(0.12, 0.35, 0.9)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = np.ones((50, 50, 3), np.float32)
    img[:, :] = demo_color
    cv2.imshow("color", img)
    cv2.waitKey()
    save_path = f"{save_dir}/color_" + "{}" + ".jpg"

    cv2.imwrite(save_path.format("origin"), img)

    # start
    density = 0
    while density <= 2.0:
        ret = img.copy()
        for e in np.nditer(ret, op_flags=["readwrite"]):
            e[...] = np.uint8(tools.ColorGrad(e, density) * 255)
        cv2.imwrite(save_path.format(density), ret)
        density = density + 0.25


def TestNoiseGauss(save_dir, scale_list=[0.1, 0.5, 1.0, 3.0]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # img = np.random.normal(0.5, scale_list[1], (1080, 720))
    save_path = f"{save_dir}/gauss_resize3_" + "{}" + ".jpg"

    # cv2.imwrite(save_path.format("origin"), img)
    # int_scale = [5, 10, 50, 100]
    for scale in scale_list:
        img = np.random.normal(0.5, scale, (360, 240))
        for e in np.nditer(img, op_flags=["readwrite"]):
            e[...] = e * 255
        ret = cv2.resize(img, (720, 1080))
        cv2.imwrite(save_path.format(scale), ret)


def TestNoiseGauss(save_dir, scale_list=[0.1, 0.5, 1.0, 3.0]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # img = np.random.normal(0.5, scale_list[1], (1080, 720))
    save_path = f"{save_dir}/gauss_resize3_" + "{}" + ".jpg"

    # cv2.imwrite(save_path.format("origin"), img)
    # int_scale = [5, 10, 50, 100]
    for scale in scale_list:
        img = np.random.normal(0.5, scale, (360, 240))
        for e in np.nditer(img, op_flags=["readwrite"]):
            e[...] = e * 255
        ret = cv2.resize(img, (720, 1080))
        cv2.imwrite(save_path.format(scale), ret)


def TestNoisePerlin(save_dir, scale_list=[3, 6, 12, 36, 72, 120]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/perlin_" + "{}" + ".jpg"

    # cv2.imwrite(save_path.format("origin"), img)
    # int_scale = [5, 10, 50, 100]
    for scale in scale_list:
        ret = tools.GetPerlinNoise(res=(scale, scale))
        for e in np.nditer(ret, op_flags=["readwrite"]):
            e[...] = e * 255
        cv2.imwrite(save_path.format(scale), ret)
        print(f"finish Perlin {scale}")


# def TestPipeline(input_path, out_dir):
#     img = tools.LoadImage(input_path)
