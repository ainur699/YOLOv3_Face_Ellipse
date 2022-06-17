from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from models import YoloV3FaceTiny
import dataset
import glob
from tqdm import tqdm
import os

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('source', None, 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

flags.DEFINE_string('weights', './checkpoints/model_best.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 512, 'resize images to')

#instead of
#dataset.DrawOutputs(img[0], outputs[0], filename, (pad[0].numpy(), pad[1].numpy()), max_shape.numpy())
def SaveEllipse(name, ellipses, pad, max_shape):
    if len(ellipses) == 0 or ellipses[0][2] == 0:
        return
    
    f = open(name, 'w')

    xy = max_shape * ellipses[0][0:2] - pad
    wh = max_shape * ellipses[0][2:4]

    center = np.array([xy[0], xy[1]])
    axes = np.array([wh[0], wh[1]])
    angle = ellipses[0][4]

    top    = np.array([center[0] + np.sin(angle) * axes[1], center[1] - np.cos(angle) * axes[1]])
    bottom = np.array([center[0] - np.sin(angle) * axes[1], center[1] + np.cos(angle) * axes[1]])
    quartet_top = bottom + 0.75 * (top - bottom)
    length = np.linalg.norm(quartet_top - bottom)

    dir = np.array([np.cos(angle), np.sin(angle)])
    bbox_tl = quartet_top - 0.5 * length * dir
    bbox_tr = quartet_top + 0.5 * length * dir
    bbox_br = bottom + 0.5 * length * dir

    scale = 1.25
    bbox_center = 0.5 * (bbox_tl + bbox_br)
    bbox_tl = bbox_center + scale * (bbox_tl - bbox_center)
    bbox_tr = bbox_center + scale * (bbox_tr - bbox_center)
    bbox_br = bbox_center + scale * (bbox_br - bbox_center)

    f.write(str(bbox_tl[0]) + ',')
    f.write(str(bbox_tl[1]) + ',')
    f.write(str(bbox_tr[0]) + ',')
    f.write(str(bbox_tr[1]) + ',')
    f.write(str(bbox_br[0]) + ',')
    f.write(str(bbox_br[1]) + '\n')
    f.close()

# Detect ellipses in images from folder imgs. Output results in bbox folder near imgs.
def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3FaceTiny(FLAGS.size)
    yolo.load_weights(FLAGS.weights).expect_partial()

    filenames = glob.glob(os.path.join(FLAGS.source, '*'))

    result_dir = os.path.join(os.path.dirname(FLAGS.source), 'bbox')
    os.makedirs(result_dir, exist_ok=True)

    for filename in tqdm(filenames):
        img, pad, max_shape = dataset.load_and_preprocess_image(filename)
        img = tf.expand_dims(img, 0)

        outputs = yolo(img)

        out_name  = os.path.join(result_dir, os.path.splitext(os.path.basename(filename))[0] + '.txt')

        SaveEllipse(out_name, outputs[0].numpy(), (pad[0].numpy(), pad[1].numpy()), max_shape.numpy())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
