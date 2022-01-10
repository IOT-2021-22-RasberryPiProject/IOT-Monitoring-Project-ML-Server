import torch


class AttributeDict(dict):
    """
    Class allowing access dictionary items as properties. 
    For example:
    
    my_dict = AttributeDict({'name': 'Adam', 'salary': 30})
    print(my_dict.name)
    >> Adam
    print(my_dict.salary)
    >> 30
    my_dict.profession = 'Worker'
    print(my_dict.profession)
    >> Worker

    """
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


config_data = {
    # mqtt params
    'broker_ip': '192.168.0.192',
    'topic': 'monitoring/frame',

    # frame params
    'frame_size': (240, 240),
    'bgr_classifier': False,

    # mediapipe detection params
    'model_selection': 0,
    'min_detection_confidence': 0.5,

    # torch classification params
    'pretrained_model': 'casia-webface',
    'torch_model_image_shape': (84, 84),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # deepnet detector params
    'detector_backend': 'ssd',
    # definitely not to use:
    # * OpenFace - tragedy
    # * OpenID - tragedy
    # * Dlib - tragedy
    # * DeepFace - tragedy
    # consider:
    # * ArcFace  - quick and slight accurate
    # * VGG-Face - best but slow
    # * Facenet - ok
    # * Facenet512 - same speed as Facenet, but better performance
    'face_detection_model': 'Facenet512', 
    'face_comparison_threshold': 0.6,
    'min_face_width': 0,

    # decision model params

    'frames_detected_threshold': 5,
    'frames_undetected_threshold': 5,
}


CONFIG = AttributeDict(config_data)

if __name__ == '__main__':
    a = AttributeDict({'Test1': 'nic'})
    a['Test2'] = 'woda'
    a.Test3= 20
    print(a.items())
