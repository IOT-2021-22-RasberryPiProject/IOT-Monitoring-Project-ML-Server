from typing import *

import os
import glob


def get_support_set_paths(path: str) -> Dict[str, List[str]]:
    support_set: Dict[str, List[str]] = dict()
    if os.path.isdir(path):
        people = [(f.name, f.path) for f in os.scandir(path) if f.is_dir()]
        for person_name, person_path in people:
            support_set[person_name] = glob.glob(os.path.join(person_path, '*'))
    return support_set


if __name__ == '__main__':
    ss = get_support_set_paths('/home/konrad/PythonProjects/IOT-Monitoring-Project/support_set')
    print(ss)
