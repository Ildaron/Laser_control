import os
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image
import cv2
from decode_np import Decode

import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


#K.tensorflow_backend._get_available_gpus()


def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

model_path = 'yolo4_weight.h5'
anchors_path = 'model_data/yolo4_anchors.txt'
classes_path = 'model_data/coco_classes.txt'
class_names = get_class(classes_path)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)
num_classes = len(class_names)
model_image_size = (608, 608)
conf_thresh = 0.2
nms_thresh = 0.45
yolo4_model = yolo4_body(Input(shape=(608,608,3)), num_anchors//3, num_classes)
model_path = os.path.expanduser(model_path)
#assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
yolo4_model.load_weights(model_path)
_decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

cap = cv2.VideoCapture(0)

while 1:
 ret, image = cap.read()   
 #image = cv2.imread(img)
 #image = cv2.imread('C:/Users/test.jpg')

 image, boxes, scores, classes = _decode.detect_image(image, True)
 cv2.imshow('image', image)
 key = cv2.waitKey(1) & 0xFF
 if key == ord("q"):
  break
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
#yolo4_model.close_session()
