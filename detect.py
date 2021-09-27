import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from models import (
    YoloV3Face, YoloV3FaceTiny
)
import dataset
import os
import glob
from tqdm import tqdm

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('image', 'D:/Images/04ed1c2ebffb11e38f8c0002c9dced72_6.jpg', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.png', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

flags.DEFINE_string('weights', './checkpoints/model_best.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 512, 'resize images to')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3FaceTiny(FLAGS.size)
    else:
        yolo = YoloV3Face(FLAGS.size)

    yolo.load_weights(FLAGS.weights).expect_partial()

    filenames = glob.glob('D:/test_images/imgs/*.jpg')

    for filename in tqdm(filenames):
        img, pad, max_shape = dataset.load_and_preprocess_image(filename)
        img = tf.expand_dims(img, 0)
        
        t1 = time.time()
        outputs = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        dataset.DrawOutputs(img[0], outputs[0], filename, (pad[0].numpy(), pad[1].numpy()), max_shape.numpy())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
