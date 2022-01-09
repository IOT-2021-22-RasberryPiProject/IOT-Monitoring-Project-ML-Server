from typing import *
import time
import binascii
import base64

import cv2
import numpy as np
from paho.mqtt.client import Client
import json

from config import AttributeDict
from models import AbstractClassifier

FRAME = np.zeros((224, 224, 3), np.uint8)


class Reciever:

    def __init__(self, config: AttributeDict, model: AbstractClassifier):
        self._config = config

        # mosquitto
        self._client = Client()
        self._client.on_connect = self.subscribe
        self._client.on_message = self.on_message

        self._try_connect(self._config.broker_ip)

        # model
        self._model = model
        # TODO wyjebać to XD
        self._frame = np.zeros((self._config.frame_size[0], self._config.frame_size[1], 3), np.uint8)

    def start(self):
        self._client.loop_start()
        while True:
            pass
        
            
    def subscribe(self, client, userdata, flags, rc) -> None:
        print(f"Connected: {rc}")
        self._client.subscribe(self._config.topic)

    def _decode_message(self, msg) -> Tuple[str, float, str]:
        msg_decoded = json.loads(msg.payload)
        name, _time, img = msg_decoded["device"], msg_decoded["time"], binascii.a2b_base64(msg_decoded["frame"])
        return name, _time, img  

    def on_message(self, client, userdata, msg) -> None:
        t1 = time.time()
        name, _time, img = self._decode_message(msg)

        frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), 1)
        results = self._model.predict(frame)

        print(results)
        for name, proba, bbox, in results:
            y1, x2, y2, x1 = bbox[0], bbox[1], bbox[2], bbox[3]

            frame = cv2.putText(
                frame, 
                f'{name}: {proba:.2f}', 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 200), 
                1)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            # cv2.imwrite('Dupa.jpg', frame)
        _, buffer = cv2.imencode('.jpg', frame)
        # Converting into encoded bytes
        message = {
            'time1': _time,
            'time2': time.time(),
            'frame': binascii.b2a_base64(buffer).decode()
        }
        message_encoded = json.dumps(message)
        self._client.publish('monitoring/show', message_encoded)
        

        t2 = time.time()
        print(1 / (t2 - t1))

    def _try_connect(self, broker_ip: str) -> None:
        try:
            self._client.connect(broker_ip)
        except Exception as e:
            print(e)

    
