import cv2
import numpy as np


from reciever import Reciever
from models import OpenCVSupportSet, OpenCVClassifier

from config import CONFIG


def main():
    support_set = OpenCVSupportSet({
        'jan': [
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/jan_wielgus/1.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/jan_wielgus/2.jpg'
            ],
        'weronika': [
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/weronika_welyczko/1.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/weronika_welyczko/2.jpg'
        ],
        'konrad': [
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/konrad_karanowski/1.jpg',
            '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/konrad_karanowski/2.jpg',

        ]
    }, CONFIG)
    model = OpenCVClassifier(
        support_set,
        CONFIG
    )
    reciever = Reciever(CONFIG, model)
    reciever.start()


if __name__ == '__main__':
    main()
