from typing import *

import cv2
import numpy as np
from keras.preprocessing import image

from models import IMAGE_TYPE


def preprocess_detected_face(
        img: IMAGE_TYPE,
        target_size: Iterable[int],
        grayscale: bool = False,
        enforce_detection: bool = True,
):
    """
    This function is adjusted to our detection system version of this:
    https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
    This version is important because it allows us not to detect face 2 times in a pipeline
    """
    # --------------------------
    base_img = img.copy()
    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:  # restore base image
            img = base_img.copy()

    # --------------------------

    # post-processing
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------
    # resize image to expected shape

    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))

        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if not grayscale:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                         'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # ------------------------------------------

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # ---------------------------------------------------

    # normalizing the image pixels

    img = image.img_to_array(img)
    img_pixels = np.expand_dims(img, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    # ---------------------------------------------------

    return img_pixels


def aggregate_scores_3sigmas(scores: Iterable[float]) -> float:
    """
    Aggregate scores using 3sigmas rule
    """
    std = np.std(scores)
    mean = np.mean(scores)
    new_scores = [score for score in scores if mean + 3 * std > score > mean - 3 * std]
    return np.mean(new_scores).item()
