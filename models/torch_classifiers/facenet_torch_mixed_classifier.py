from typing import *
import math

import time 
import torch
from torch import nn
import cv2
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
import mediapipe as mp
import numpy as np

from config import AttributeDict
from models.abstract_classifier import AbstractSupportSet, AbstractClassifier, IMAGE_TYPE, BOUNDING_BOX


def _get_image_transforms(target_size: Tuple[int, int], normalize: bool = True) -> transforms.Compose:
    _tr = [
        transforms.ToTensor(),
        fixed_image_standardization,
        transforms.Resize(target_size)
    ]
    return transforms.Compose(_tr)

def _cossine_similarity(embeddings1, embeddings2):
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=0)
    norm = np.linalg.norm(embeddings1, axis=0) * np.linalg.norm(embeddings2, axis=0)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return dist



class FaceNetSupportSet(AbstractSupportSet):

    def __init__(self, support_set: Dict[str, List[str]], config: AttributeDict, **kwargs) -> None:
        super(FaceNetSupportSet, self).__init__(support_set, config, **kwargs)
        self._transforms = _get_image_transforms(self._config.torch_model_image_shape)
        self._is_initalized = False

    def _get_support_set(self, support_set: Dict[str, List[str]]) -> Dict[str, List[IMAGE_TYPE]]:
        return support_set

    # TORCH SUPPORT SET REQUIRES OUTER INITIALIZATION
    def initialize_support_set(self, model: nn.Module) -> None:
        _set: Dict[str, List[IMAGE_TYPE]] = {}
        for key in self._support_set.keys():
            image_embeddings = []
            for path in self._support_set[key]:
                with torch.no_grad():
                    img = self._load_image(path)
                    embedding = model(self._transforms(img).unsqueeze(0).to(self._config.device)).view(-1).to('cpu').numpy()
                    image_embeddings.append(embedding)
            _set[key] = image_embeddings
        self._support_set = _set
        self._is_initalized = True

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img


class FaceNetClassifier(AbstractClassifier):
    
    def __init__(self, support_set: FaceNetSupportSet, config: AttributeDict, pretrained: str = 'vggface2', **kwargs):
        super(FaceNetClassifier, self).__init__(support_set, config, **kwargs)
        # Initialize torch model
        self._torch_model:  nn.Module = InceptionResnetV1(classify=False, pretrained=self._config.pretrained_model)
        self._torch_model.to(self._config.device)
        self._torch_model.eval()

        # Initialize support set IMPORTANT
        self._support_set.initialize_support_set(self._torch_model)
        self._transforms = _get_image_transforms(self._config.torch_model_image_shape)

    def _detect_faces(self, img: IMAGE_TYPE) -> List[Tuple[float, BOUNDING_BOX]]:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=self._config.model_selection, 
            min_detection_confidence=self._config.min_detection_confidence) as face_detection:
            detections = face_detection.process(img)
            # results
            results = []
            if detections.detections:
                h, w, _ = img.shape
                for detection in detections.detections:
                    box = detection.location_data.relative_bounding_box
                    x1, y1, w, h = int(abs(box.xmin) * w), int(abs(box.ymin) * h), int(abs(box.width) * w * 1.2), int(abs(box.height) * h * 1.2)
                    bounding_box = y1, x1 + w, y1 + h, x1
                    results.append(
                        (detection.score[0], bounding_box)
                    )
            return results

    def _process_image(self, img, bounding_box):
        cropped = img[bounding_box[0]:bounding_box[2], bounding_box[3]:bounding_box[1], :]
        image = self._transforms(
            cropped
        ).unsqueeze(0).to(self._config.device)
        return image

    def _process_scores(self, scores: List[Tuple[str, List[float]]]) -> Tuple[str, float]:
        means = list(
            map(lambda x: (x[0], np.mean(x[1])), scores)
        )
        best_result = min(means, key=lambda x: x[1])
        return best_result

    def _calculate_scores(self, found_emb, face_embeddings):
        similarities = []
        for emb in face_embeddings:
            similarities.append(_cossine_similarity(found_emb, emb))
        return  np.mean(similarities)

    def _recognize_faces(self, img, bounding_box: BOUNDING_BOX) -> Tuple[str, float, BOUNDING_BOX]:
        # bbox = bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]
        image = self._process_image(img, bounding_box)
        all_scores = []
        with torch.no_grad():
            for key, item in self._support_set.items():
                img_emb = self._torch_model(image).view(-1).to('cpu').numpy()
                scores = self._calculate_scores(img_emb, item)
                all_scores.append((key, scores))
            best_result = self._process_scores(all_scores)
            return best_result[0], 1 - best_result[1], bounding_box


    def predict(self, img: IMAGE_TYPE) -> List[Tuple[str, float, BOUNDING_BOX]]:
        if self._config.bgr_classifier:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        detections = self._detect_faces(img)
        print(detections)
        t2 = time.time()
        print(t2-t1)
        result = []
        for detection in detections:
            prediction = self._recognize_faces(img, detection[1])
            result.append(prediction)
        print(result)
        return result
