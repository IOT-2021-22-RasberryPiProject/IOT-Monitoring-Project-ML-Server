from typing import *
import abc

import numpy as np
from config import AttributeDict

IMAGE_TYPE = np.ndarray
BOUNDING_BOX = Tuple[int, int, int, int]


class AbstractSupportSet(abc.ABC):
    _support_set: Dict[str, List[IMAGE_TYPE]]

    def __init__(self, support_set: Dict[str, List[str]], config: AttributeDict, **kwargs) -> None:
        self._config = config
        self._support_set: Dict[str, List[IMAGE_TYPE]] = self._get_support_set(support_set)
        self._assert_support_set_size()

    def _assert_support_set_size(self) -> None:
        n_shot = len(self._support_set[next(iter(self._support_set))])
        for key in self._support_set.keys():
            candidate_n_shot = len(self._support_set[key])
            assert n_shot == candidate_n_shot, f'Every class should contain the same number of samples but got: {candidate_n_shot} and {n_shot}'

    @abc.abstractmethod
    def _get_support_set(self, support_set: Dict[str, List[str]]) -> Dict[str, List[IMAGE_TYPE]]:
        pass

    def keys(self):
        return self._support_set.keys()

    @property
    def n_way(self) -> int:
        return len(self._support_set.keys())

    @property
    def n_shot(self) -> int:
        return len(self._support_set[next(iter(self._support_set))])

    def __getitem__(self, idx: str) -> Iterable[IMAGE_TYPE]:
        return self._support_set[idx]

    def items(self):
        return self._support_set.items()


class AbstractClassifier(abc.ABC):

    def __init__(self, support_set: AbstractSupportSet or Dict[str, Iterable[Any]], config: AttributeDict, **kwargs):
        self._support_set = support_set
        self._config = config

    @property
    def people(self) -> List[str]:
        return list(self._support_set.keys())

    @abc.abstractmethod
    def predict(self, frame: IMAGE_TYPE) -> List[Tuple[str, float, BOUNDING_BOX]]:
        pass
