import keras.backend as K
import numpy as np
import tensorflow as tf
#%tensorflow_version 1.x
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from loss import yolo_loss
from yolo4_tiny import yolo_body
from utils.utils import (WarmUpCosineDecayScheduler, get_random_data,
                         get_random_data_with_Mosaic, rand)




def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False, random=True):
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i+4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i+4], input_shape)
                    i = (i+4) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                    i = (i+1) % n
                flag = bool(1-flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                i = (i+1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'

    num_layers = len(anchors)//3

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]


    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32') 

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]


    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]

    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]


    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes


    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
       
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue

        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes


        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):

            for l in range(num_layers):
                if n in anchor_mask[l]:

                    i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')

                    k = anchor_mask[l].index(n)

                    c = true_boxes[b, t, 4].astype('int32')

                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


if __name__ == "__main__":

    annotation_path = '/content/drive/MyDrive/yolov4_tiny/yolov4-tiny-keras-master/new_train.txt'

    log_dir = 'logs/'

    classes_path = '/content/drive/MyDrive/yolov4_tiny/yolov4-tiny-keras-master/voc_classes.txt'    
    anchors_path = '/content/drive/MyDrive/yolov4_tiny/yolov4-tiny-keras-master/yolo_anchors.txt'

    weights_path = '/content/drive/MyDrive/yolov4_tiny/yolov4-tiny-keras-master/yolov4_tiny_weights_voc.h5'

    input_shape = (416,416)

    normalize = False

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)
    mosaic = False
    Cosine_scheduler = False
    label_smoothing = 0

    K.clear_session()

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv4-Tiny model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors//2, num_classes)
    

    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], num_anchors//2, num_classes+5)) for l in range(2)]
    loss_input = [*model_body.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'label_smoothing': label_smoothing, 'normalize':normalize})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)


    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    

    freeze_layers = 60
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

 
    if True:
        Init_epoch = 0
        Freeze_epoch = 50
        batch_size = 32
        learning_rate_base = 1e-3

        if Cosine_scheduler:
            
            warmup_epoch = int((Freeze_epoch-Init_epoch)*0.2)
          
            total_steps = int((Freeze_epoch-Init_epoch) * num_train / batch_size)
           
            warmup_steps = int(warmup_epoch * num_train / batch_size)
          
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                        total_steps=total_steps,
                                                        warmup_learning_rate=1e-4,
                                                        warmup_steps=warmup_steps,
                                                        hold_base_rate_steps=num_train,
                                                        min_learn_rate=1e-6
                                                        )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False),
                validation_steps=max(1, num_val//batch_size),
                epochs=Freeze_epoch,
                initial_epoch=Init_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    for i in range(freeze_layers): model_body.layers[i].trainable = True

   
    if True:
        Freeze_epoch = 50
        Epoch = 100
        batch_size = 16
        learning_rate_base = 1e-4

        if Cosine_scheduler:
     
            warmup_epoch = int((Epoch-Freeze_epoch)*0.2)
           
            total_steps = int((Epoch-Freeze_epoch) * num_train / batch_size)
        
            warmup_steps = int(warmup_epoch * num_train / batch_size)
   
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                        total_steps=total_steps,
                                                        warmup_learning_rate=1e-5,
                                                        warmup_steps=warmup_steps,
                                                        hold_base_rate_steps=num_train//2,
                                                        min_learn_rate=1e-6
                                                        )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False),
                validation_steps=max(1, num_val//batch_size),
                epochs=Epoch,
                initial_epoch=Freeze_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'last1.h5')
