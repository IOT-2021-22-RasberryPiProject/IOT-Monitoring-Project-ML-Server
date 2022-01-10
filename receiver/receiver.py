from typing import *
import time
import binascii

import cv2
import numpy as np
from paho.mqtt.client import Client
import json

from config import AttributeDict
from models import AbstractClassifier
from decision_models import DecisionHandler, DecisionModel

FRAME = np.zeros((224, 224, 3), np.uint8)


class Receiver:

    def __init__(self, config: AttributeDict, model: AbstractClassifier):
        self._config = config

        # decision model
        print(model.people)
        self._decision_model = DecisionModel(config, model.people)
        self._decision_handler = DecisionHandler(config, model.people)

        # mosquitto
        self._client = Client()
        self._client.on_connect = self.subscribe
        self._client.on_message = self.on_message

        self._try_connect(self._config.broker_ip)

        # model
        self._model = model

        self._frame = FRAME

    def start(self):
        self._client.loop_start()
        while True:
            cv2.imshow('Frame', self._frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break

    def stop(self):
        pass

    def subscribe(self, client, userdata, flags, rc) -> None:
        print(f"Connected: {rc}")
        self._client.subscribe(self._config.topic)

    def _decode_message(self, msg) -> Tuple[str, float, bytes]:
        msg_decoded = json.loads(msg.payload)
        name, _time, img = msg_decoded["device"], msg_decoded["time"], binascii.a2b_base64(msg_decoded["frame"])
        return name, _time, img

    def on_message(self, client, userdata, msg) -> None:
        name, _time, img = self._decode_message(msg)

        frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), 1)
        results = self._model.predict(frame)

        for name, proba, bbox, in results:
            x, y, w, h = bbox

            frame = cv2.putText(
                frame,
                f'{name}:{round(proba * 100, 2)}%',
                (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 200),
                1)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 4)
        self._frame = frame
        successes = self._decision_model.verify_decision(results)
        self._decision_handler.handle_detections(successes)

    def _try_connect(self, broker_ip: str) -> None:
        try:
            self._client.connect(broker_ip)
        except Exception as e:
            print(e)