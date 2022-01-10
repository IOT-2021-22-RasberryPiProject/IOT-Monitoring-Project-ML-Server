from typing import *
import abc

import numpy as np
from config import AttributeDict

# simple typedefs
IMAGE_TYPE = np.ndarray
BOUNDING_BOX = Tuple[int, int, int, int]


class AbstractSupportSet(abc.ABC):
    """
    Abstract class representing support set in few-shot learning
    (check this videos: https://www.youtube.com/watch?v=hE7eGew4eeg)
    """

    def __init__(self, support_set: Dict[str, List[str]], config: AttributeDict, **kwargs) -> None:
        self._config = config
        self._support_set: Dict[str, List[IMAGE_TYPE]] = self._get_support_set(support_set)

    @abc.abstractmethod
    def _get_support_set(self, support_set: Dict[str, List[str]]) -> Dict[str, List[IMAGE_TYPE]]:
        """
        Parse support set
        """
        pass

    @property
    def n_way(self) -> int:
        """
        Get n-way (how many classes)
        """
        return len(self._support_set.keys())

    @property
    def n_shot(self) -> int:
        """
        Get n-shot (how many examples of one class)
        """
        return len(self._support_set[next(iter(self._support_set))])

    def __getitem__(self, key: str) -> Iterable[IMAGE_TYPE]:
        """
        Get item from support set
        """
        return self._support_set[key]

    def keys(self):
        """
        Return keys
        """
        return self._support_set.keys()

    def items(self):
        """
        Return items
        """
        return self._support_set.items()


class AbstractClassifier(abc.ABC):
    """
    Abstract class representing Few-Shot classifier
    (check this videos: https://www.youtube.com/watch?v=hE7eGew4eeg)
    """

    def __init__(self, support_set: AbstractSupportSet or Dict[str, Iterable[Any]], config: AttributeDict, **kwargs):
        self._support_set = support_set
        self._config = config

    @property
    def people(self) -> List[str]:
        """
        List of people to detect
        """
        return list(self._support_set.keys())

    @abc.abstractmethod
    def predict(self, frame: IMAGE_TYPE) -> List[Tuple[str, float, BOUNDING_BOX]]:
        """
        Make a prediction
        """
        pass
