import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from models import (
    YoloV3, YoloV3Tiny, YoloV3Face
)
import dataset
import os
import glob

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_face2_train_16.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 512, 'resize images to')
flags.DEFINE_string('image', 'D:/Images/04ed1c2ebffb11e38f8c0002c9dced72_6.jpg', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.png', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)


    yolo = YoloV3Face(FLAGS.size)
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

#    for filename in os.listdir('D:/Images'):
#        if filename.endswith(".jpg"):
#            img = dataset.load_and_preprocess_image('D:/Images/' + filename)
#            img = tf.expand_dims(img, 0)
#
#            t1 = time.time()
#            outputs = yolo(img)
#            t2 = time.time()
#            logging.info('time: {}'.format(t2 - t1))
#
#            dataset.DrawOutputs(img, outputs, './results/' + filename)

    root_path = 'D:/Datasets/FDDB'
    label_files = glob.glob(os.path.join(root_path, 'FDDB-folds\\FDDB-fold-*-ellipseList.txt'))

    img_num = 0

    for fname in label_files:
        if img_num > 100: break
        with open(fname) as f:
            raw_label = f.read().strip().split('\n')
            
            id = 0
            
            while id < len(raw_label):
                image_name = os.path.join(root_path, 'originalPics', raw_label[id]) + '.jpg'
                image_name = os.path.normpath(image_name)
                ell_num = int(raw_label[id + 1])
                id += 2
                
                ellipses = []
                for i in range(ell_num):
                    ell = tuple(map(float, raw_label[id+i].split(' ')[:-2]))
                    ellipses.append(ell)
                
                id += ell_num
                
                img, _, _ = dataset.load_and_preprocess_image(image_name)
                img = tf.expand_dims(img, 0)
                outputs = yolo(img)
                img_num = 1 + img_num
                dataset.DrawOutputs(img[0], outputs[0], './results/' + os.path.basename(image_name))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
