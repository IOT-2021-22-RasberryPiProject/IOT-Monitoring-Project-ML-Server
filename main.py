from receiver import Receiver
from models import DeepFaceClassifier, get_support_set_paths

from config import CONFIG

PATH = '/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set'


def main() -> None:
    model = DeepFaceClassifier(get_support_set_paths(PATH), CONFIG)
    receiver = Receiver(CONFIG, model)
    receiver.start()


if __name__ == '__main__':
    main()
