from typing import *
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re
from keras.preprocessing import image
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector
# from abstract_classifier import AbstractClassifier, AbstractSupportSet, BOUNDING_BOX, IMAGE_TYPE
from config import AttributeDict, CONFIG


def preprocess_detected_face(
        img,
        target_size,
        grayscale=False,
        enforce_detection=True,
):
    # --------------------------
    base_img = img.copy()
    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection == True:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:  # restore base image
            img = base_img.copy()

    # --------------------------

    # post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------
    # resize image to expected shape

    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
    i0s = img.shape
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))

        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                         'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # ------------------------------------------

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        try:
            img = cv2.resize(img, target_size)
        except:
            print(f'Img0 shape: {i0s}')
            print(f'Wyjebało się, img: {img.shape}, dsize: {target_size}')

    # ---------------------------------------------------

    # normalizing the image pixels

    img = image.img_to_array(img)
    img_pixels = np.expand_dims(img, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    # ---------------------------------------------------

    return img_pixels




class DeepFaceClassifier:

    """
    Pipeline:
    PRE_CLASSIFY:
        1. build model
        2. build embeddings
    CLASSIFY:
    """

    def __init__(self, support_set_paths: Dict[str, Iterable[str]], config: AttributeDict) -> None:
        self._config = config
        
        # build face detector model
        self._face_detector_model = FaceDetector.build_model(self._config.detector_backend)

        # build face comparison model
        self._face_compare_model = DeepFace.build_model(self._config.face_detection_model)
        self._shape = functions.find_input_shape(self._face_compare_model)[::-1]

        self._support_set = self._build_embeddings(support_set_paths)

    def _build_embeddings(self, support_set_paths: Dict[str, Iterable[str]]):
        _support_set = {}
        for person, paths in support_set_paths.items():
            embeddings = []
            for image_path in paths:
                try:
                    img = functions.preprocess_face(
                        img=image_path, 
                        target_size=self._shape, 
                        enforce_detection=True, 
                        detector_backend=self._config.detector_backend
                    )
                    # plt.suptitle('Embedding face')
                    # plt.imshow(img[0, :])
                    # plt.show()
                    embedding = self._face_compare_model.predict(img)[0, :]
                    embeddings.append(embedding)
                except:
                    print(f"Couldn't find face for photo: {image_path}")
            _support_set[person] = embeddings
        return _support_set


    def _aggregate_scores(self, scores):
        std = np.std(scores)
        mean = np.mean(scores)
        new_scores = [score for score in scores if score < mean + 3 * std and score > mean - 3 * std]
        return np.mean(new_scores)

    def _get_score(self, img_embedding):
        keys = list(self._support_set.keys())
        distances = []
        for key in keys:
            scores = []
            for embedding in self._support_set[key]:
                score = dst.findCosineDistance(img_embedding, embedding)
                scores.append(score)
            mean_distance = self._aggregate_scores(scores)
            distances.append(mean_distance)
        best_idx = np.argmin(distances)
        # IMPORTANT! DECISION PROCESS 
        if distances[best_idx] > self._config.face_comparison_threshold:
            return 'Undefined', distances[best_idx]
        else:
            return keys[best_idx], distances[best_idx]

    def predict(self, img) -> None:
        # detect faces 
        try:
            faces = FaceDetector.detect_faces(
                self._face_detector_model,
                self._config.detector_backend,
                img,
                align=False
            )
        except:
            faces = []
        
        detected = []
        for face, (x, y, w, h) in faces:
            if w > self._config.min_face_width:
                # plt.suptitle('Detected face')
                # plt.imshow(face)
                # plt.show()
                if face.shape[0] != 0:
                    detected.append((
                        face, (x, y, w, h)
                    ))

        scores = []
        if detected:
            for face, bbox in detected[:1]:
                preprocessed_image = preprocess_detected_face(
                    face, 
                    target_size=self._shape,
                    enforce_detection=False
                )
                # plt.suptitle('Processed face')
                # plt.imshow(preprocessed_image[0, :])
                # plt.show()
                img_embedding = self._face_compare_model.predict(preprocessed_image)[0, :]
                name, score = self._get_score(img_embedding)
                scores.append((
                    name, 1 - score, bbox
                ))
        return scores



if __name__ == '__main__':
    d = DeepFaceClassifier({
        'Natalia': [
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Natalia_Psiuk/1.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Natalia_Psiuk/2.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Natalia_Psiuk/3.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Natalia_Psiuk/4.jpg'
        ],
        'Michaś': [
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Michał_Żądełek/1.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Michał_Żądełek/2.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Michał_Żądełek/3.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Michał_Żądełek/4.jpg'
        ]
    }, CONFIG)
    img = cv2.imread('/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Weronika_Welyczko/1.jpg')
    res = d.predict(img)
    print(res)


    cap = cv2.VideoCapture(0) 
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (112, 112))

        f1 = time.time()
        preds = d.predict(frame)
        print(preds)
        f2 = time.time()
        print(f'FPS: {1/(f2-f1)}')

        for name, proba, bbox, in preds:
            x, y, w, h = bbox

            frame = cv2.putText(
                frame, 
                f'{name}: {proba:.2f}', 
                (x, y + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 200), 
                1)
            frame = cv2.rectangle(frame, (x, y), (y * 2 + h, x + w), (0, 0, 200), 4)
        cv2.imshow("Stream", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break