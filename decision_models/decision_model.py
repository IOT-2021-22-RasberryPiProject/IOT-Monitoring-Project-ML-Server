from typing import *
import time


from models import BOUNDING_BOX
from config import AttributeDict


"""
Decision process: 
1. Determine if person is detected -> DecisionModel
---------------------------------------
START:
    people = {person: {frames_detected, frames_undetected, status}

FUNC (detections):
    successes = []
    FOR person in people:
        IF person in detections:
            IF not person.detected:
                person.frames_detected--
                IF person.frames_detected <= 0:
                    person.status = detected
                    person.frames_detected = THRESHOLD_DETECTED
            ELSE:
                person.frames_undetected = 0
        ELSE:
            IF person.detected:
                person.frames_undetected--
                IF person.frames_undetected <= 0:
                    person.status = undetected
                    person.frames_undetected = THRESHOLD_UNDETECTED
                    successes.append(frames)
            ELSE: 
                person.frames_detected = 0
    RETURN successes
END;

2. Switch status of person detection -> DecisionHandler
---------------------------------------
FUNC (successes):
    FOR person in successes:
        switch person state
        DO something
END;
"""


class DecisionModel:

    class _Person:

        def __init__(self, name: str, frames_undetected: int, frames_detected: int, detected: bool = False):
            self.name = name
            self.frames_undetected = frames_undetected
            self.frames_detected = frames_detected
            self.detected = detected

        def __repr__(self) -> str:
            return f'{self.name}, frames detected: {self.frames_detected}, undet: {self.frames_undetected}, detected: {self.detected}'

    def __init__(self, config: AttributeDict, people: List[str]) -> None:
        self._config = config
        # setup people
        self._people = [
            self._Person(
                name=person_name,
                frames_detected=self._config.frames_detected_threshold,
                frames_undetected=self._config.frames_undetected_threshold,
                detected=False
            ) for person_name in people
        ]

    def verify_decision(self, detections: List[Tuple[str, float, BOUNDING_BOX]]) -> List[str]:
        detections = map(lambda x: x[0], detections)
        successes = []
        for person in self._people:
            if person.name in detections:
                if not person.detected:
                    # if person is not detected, we are closer to detecting him
                    person.frames_detected -= 1
                    # if he reaches threshold we detected him
                    if person.frames_detected <= 0:
                        person.detected = True
                        person.frames_detected = self._config.frames_detected_threshold
                else:
                    person.frames_undetected = self._config.frames_undetected_threshold
            else:
                if person.detected:
                    person.frames_undetected -= 1
                    if person.frames_undetected <= 0:
                        person.detected = False
                        person.frames_undetected = self._config.frames_undetected_threshold
                        successes.append(person.name)
                else:
                    person.frames_detected = self._config.frames_detected_threshold
        return successes


class DecisionHandler:
    """
    Assert every person is OUTSIDE
    """

    def __init__(self, config: AttributeDict, people: List[str]):
        self._config = config
        self._people = {
            person: False # False == outside
            for person in people
        }

    def handle_detections(self, successes: List[str]) -> None:
        for person in successes:
            self._people[person] = not self._people[person]
            if self._people[person]:
                print(f'{person} came inside {time.strftime("%H:%M:S", time.gmtime())}')
            else:
                print(f'{person} came outside {time.strftime("%H:%M:S", time.gmtime())}')
