from typing import *

import time 
import cv2
import face_recognition
import numpy as np

from config import AttributeDict
from models.abstract_classifier import AbstractSupportSet, AbstractClassifier, IMAGE_TYPE, BOUNDING_BOX




class OpenCVSupportSet(AbstractSupportSet):
    
    def __init__(self, support_set: Dict[str, List[str]], config: AttributeDict, **kwargs) -> None:
        super(OpenCVSupportSet, self).__init__(support_set, config, **kwargs)

    def _get_support_set(self, support_set: Dict[str, List[str]]) -> Dict[str, List[IMAGE_TYPE]]:
        _set: Dict[str, List[IMAGE_TYPE]] = {}
        for key in support_set.keys():
            image_embeddings = []
            for path in support_set[key]:
                try:
                    img = self._load_image(path)
                    embedding = self._get_embedding(img)
                    image_embeddings.append(embedding)
                except IndexError as e:
                    print(f"Couldn't recognize face at image: {path}, consider other image")
            _set[key] = image_embeddings
        return _set

    def _load_image(self, path: str) -> np.ndarray:
            img = cv2.imread(path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return rgb_img

    def _get_embedding(self, img: np.ndarray) -> np.ndarray:
        embedding = face_recognition.face_encodings(img)[0]
        return embedding


class OpenCVClassifier(AbstractClassifier):
    
    def __init__(self, support_set: OpenCVSupportSet, config: AttributeDict, **kwargs):
        super(OpenCVClassifier, self).__init__(support_set, config, **kwargs)
        self.detector = cv2.CascadeClassifier('/home/konrad/PythonProjects/IOT-Monitoring-Project/files/haar.xml')

    def _compute_score(self, face_embeddings: List[np.ndarray], captured_face: IMAGE_TYPE) -> List[float]:
        return face_recognition.face_distance(face_embeddings, captured_face)

    def _process_scores(self, scores: List[Tuple[str, List[float]]]) -> Tuple[str, float]:
        means = list(
            map(lambda x: (x[0], np.mean(x[1])), scores)
        )
        best_result = min(means, key=lambda x: x[1])
        return best_result

    def predict(self, img: IMAGE_TYPE) -> List[Tuple[str, float, BOUNDING_BOX]]:
        # convert BGR2RGB may be redundant in the future!
        if self._config.bgr_classifier:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t1 = time.time()
        rects = self.detector.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30))
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        t2 = time.time()
        face_encodings = face_recognition.face_encodings(img, face_locations)
        t3 = time.time()

        # result
        result: List[Tuple[str, float, BOUNDING_BOX]] = []
        for bounding_box, face_encoding in zip(face_locations, face_encodings):
            all_scores = []
            for key, item in self._support_set.items():
                scores = self._compute_score(item, face_encoding)
                all_scores.append((key, scores))
            t4 = time.time()
            best_result = self._process_scores(all_scores)
            result.append(
                (best_result[0], 1 - best_result[1], bounding_box)
            )
            print(f'-> {t2 - t1}, 2 ->{t3 -t2}, 3 -> {t4-t3}')
        return result


if __name__ == '__main__':
    support_set = OpenCVSupportSet({
        'jan': [
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/jan_wielgus/1.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/jan_wielgus/2.jpg'
            ],
        'weronika': [
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/weronika_welyczko/1.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/weronika_welyczko/2.jpg'
        ]
    }, None)

    classifier = OpenCVClassifier(support_set, None)
    print(classifier.predict(cv2.imread('/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/jan_wielgus/1.jpg')))
