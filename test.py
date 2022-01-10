from config import CONFIG
from models import DeepFaceClassifier
import cv2
import time

from decision_models.decision_model import DecisionModel, DecisionHandler

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
    model = DecisionModel(CONFIG, ['Natalia', 'Michaś', 'Undefined'])
    dh = DecisionHandler(CONFIG, ['Natalia', 'Michaś', 'Undefined'])
    img = cv2.imread('/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set/Weronika_Welyczko/1.jpg')
    res = d.predict(img)
    print(res)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))

        f1 = time.time()
        preds = d.predict(frame)
        sucs = model.verify_decision(preds)
        dh.handle_detections(sucs)
        #print('Successes: ', sucs)
        #print(preds)
        # time.sleep(0.5)
        f2 = time.time()
        #print(f'FPS: {1 / (f2 - f1)}')

        for name, proba, bbox, in preds:
            x, y, w, h = bbox

            frame = cv2.putText(
                frame,
                f'{name}: {round(proba, 4) * 100}',
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 200),
                1)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 4)
        cv2.imshow("Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
