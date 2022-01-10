from receiver import Receiver
from models import DeepFaceClassifier

from config import CONFIG


def main() -> None:
    model = DeepFaceClassifier({
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
    receiver = Receiver(CONFIG, model)
    receiver.start()


if __name__ == '__main__':
    main()
