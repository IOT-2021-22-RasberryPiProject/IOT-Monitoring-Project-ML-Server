from typing import *

import time
import os
from tqdm import tqdm

import numpy as np
from deepface import DeepFace
from deepface.commons import functions, distance as dst
from deepface.detectors import FaceDetector

from config import AttributeDict
from models.abstract_classifier import AbstractClassifier, BOUNDING_BOX, IMAGE_TYPE
from models.deepface_classifiers.utils import preprocess_detected_face, aggregate_scores_3sigmas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepFaceClassifier(AbstractClassifier):
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
        for person, paths in tqdm(support_set_paths.items()):
            embeddings = []
            for image_path in paths:
                try:
                    img = functions.preprocess_face(
                        img=image_path,
                        target_size=self._shape,
                        enforce_detection=True,
                        detector_backend=self._config.detector_backend
                    )
                    embedding = self._face_compare_model.predict(img)[0, :]
                    embeddings.append(embedding)
                except:
                    print(f"Couldn't find face for photo: {image_path}")
            _support_set[person] = embeddings
        return _support_set

    def _get_score(self, img_embedding) -> Tuple[str, float]:
        keys = list(self._support_set.keys())
        distances = []
        for key in keys:
            scores = []
            for embedding in self._support_set[key]:
                score = dst.findCosineDistance(img_embedding, embedding)
                scores.append(score)
            mean_distance = aggregate_scores_3sigmas(scores)
            distances.append(mean_distance)
        best_idx = np.argmin(distances)
        # IMPORTANT! DECISION PROCESS 
        if distances[best_idx] > self._config.face_comparison_threshold:
            return 'Undefined', distances[best_idx]
        else:
            return keys[best_idx], distances[best_idx]

    def predict(self, img: IMAGE_TYPE) -> List[Tuple[str, float, BOUNDING_BOX]]:
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
                img_embedding = self._face_compare_model.predict(preprocessed_image)[0, :]
                name, score = self._get_score(img_embedding)
                scores.append((
                    name, 1 - score, bbox
                ))
        return scores
