import cv2
from util.reid_onnx_helper import ReidHelper
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache


@lru_cache(maxsize=12000)
def get_images(img_path):
    return cv2.imread(img_path)


def calc_distance(feat1, feat2):
    distance = np.linalg.norm(feat1 - feat2)
    return distance


def calc_cosign_similarity(feat1, feat2):
    cos_sim = np.squeeze(feat1 @ feat2.T)
    return cos_sim

def sample_distance(reid, plot=False, verbose=1):

    helper = ReidHelper(reid)

    BASE_FOLDER = "sample"

    feat1 = None
    min_distance = 0
    min_file = None

    files = os.listdir(BASE_FOLDER)

    h = len(files)//5
    w = 5
    if plot:
        fig, axs = plt.subplots(h, w, figsize=(15, 7))

    for index, file in enumerate(files):
        file_path = os.path.join(BASE_FOLDER, file)
        car_img = cv2.imread(file_path)

        feat = helper.infer(car_img)
        if index == 0:
            feat1 = feat
            if plot:
                axs[index, index].imshow(car_img)
                axs[index, index].set_title("target_image")
        else:
            distance = calc_distance(feat1, feat)
            if plot:
                axs[index//w, index % w].imshow(car_img)
                axs[index//w, index % w].set_title(str(distance))
            if verbose > 0:
                print(file, distance)

            if min_distance == 0 or min_distance > distance:
                min_distance = distance
                min_file = file
        if plot:
            axs[index//w, index % w].set_xticks([])
            axs[index//w, index % w].set_yticks([])

    if plot:
        fig.tight_layout()
        plt.show()
    if verbose > 0:
        print("target_file:", files[0])
        print("min_file:", min_file)


def sample_similarity(reid, plot=False, verbose=1):

    helper = ReidHelper(reid)

    BASE_FOLDER = "sample"

    feat1 = None
    max_sim = 0
    max_file = None

    files = os.listdir(BASE_FOLDER)

    h = len(files)//5
    w = 5

    if plot:
        fig, axs = plt.subplots(h, w, figsize=(15, 7))

    for index, file in enumerate(files):
        file_path = os.path.join(BASE_FOLDER, file)
        car_img = cv2.imread(file_path)

        feat = helper.infer(car_img)
        if index == 0:
            feat1 = feat
            if plot:
                axs[index, index].imshow(car_img)
                axs[index, index].set_title("target_image")
        else:
            sim = calc_cosign_similarity(feat1, feat)
            if plot:
                axs[index//w, index % w].imshow(car_img)
                axs[index//w, index % w].set_title(str(sim))
            
            if verbose > 0:
                print(file, sim)

            if max_sim == 0 or max_sim < sim:
                max_sim = sim
                max_file = file
        if plot:
            axs[index//w, index % w].set_xticks([])
            axs[index//w, index % w].set_yticks([])
    if plot:
        fig.tight_layout()
        plt.show()
    if verbose > 0:
        print("target_file:", files[0])
        print("max_file:", max_file)
