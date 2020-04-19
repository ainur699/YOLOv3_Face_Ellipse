from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import cv2

import glob
import os


def LoadFDDB(root_path):
    label_files = glob.glob(os.path.join(root_path, 'FDDB-folds\\FDDB-fold-*-ellipseList.txt'))

    data = {}
    data['images'] = []
    data['ellipses'] = []

    for fname in label_files:
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
                
                data['images'].append(image_name)
                data['ellipses'].append(ellipses)
    return data


def DrawExample(example):
    image = 255.0 * example[0].numpy()
    ellipses = example[1].numpy()
    
    for ell in ellipses:
        center_coordinates = (int(ell[3]), int(ell[4]))
        axesLength = (int(ell[0]), int(ell[1]))
        angle = int(180.0 / 3.1416 * ell[2])
        angle = angle - 90 if angle >= 0 else angle + 90 

        cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0,255,0), 1)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('debug.png', image)


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255.0
    shape = tf.shape(image)
    image = tf.pad(image, [[0, 450 - shape[0]], [0, 450 - shape[1]], [0, 0]])
    image = tf.image.resize(image, (FLAGS.size, FLAGS.size))

    return image

def preprocess_label(y):
    ratio = FLAGS.size / 450.0;
    ratio = [ratio, ratio, 1, ratio, ratio]
    ratio = tf.cast(ratio, tf.float32)
    y = tf.multiply(y, ratio)

    angle = y[..., 2];
    angle = tf.where(angle > 0, angle-1.57, angle+1.57)
    angle = tf.expand_dims(angle, axis=-1)
    y = tf.concat([y[..., 0:2], angle, y[..., 3:5]], axis=-1)  

    return y;

def CreateFDDB(root_path):
    data = LoadFDDB(root_path)

    TRAIN_LENGTH = int(0.8 * len(data['images']))
    TEST_LENGTH = len(data['images']) - TRAIN_LENGTH
    BUFFER_SIZE = 1000

    labels = tf.data.Dataset.from_generator(lambda: data['ellipses'], tf.float32, [None, 5])
    labels = labels.map(preprocess_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    images = tf.data.Dataset.from_tensor_slices(data['images'])
    images = images.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = tf.data.Dataset.zip((images, labels))

    # todo center data

    ds = ds.shuffle(BUFFER_SIZE)

    train = ds.take(TRAIN_LENGTH)
    test = ds.skip(TEST_LENGTH)

    return (train, test)


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]

    box_wh = y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]

    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (ax0, ax1, x, y, angle, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, angle, obj])
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][0], 0):
                continue

            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                ell = y_true[i][j][0:5]
                box_xy = y_true[i][j][3:4]

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, angle, obj)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [ell[3], ell[4], ell[0], ell[1], ell[2], 1])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


## https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
## Commented out fields are not required in our project
#IMAGE_FEATURE_MAP = {
#    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
#    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
#    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
#    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
#    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
#    'image/encoded': tf.io.FixedLenFeature([], tf.string),
#    # 'image/format': tf.io.FixedLenFeature([], tf.string),
#    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
#    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
#    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
#    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
#    'image/object/class/text': tf.io.VarLenFeature(tf.string),
#    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
#    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
#    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
#    # 'image/object/view': tf.io.VarLenFeature(tf.string),
#}


#def parse_tfrecord(tfrecord, class_table, size):
#    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
#    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
#    x_train = tf.image.resize(x_train, (size, size))
#
#    class_text = tf.sparse.to_dense(
#        x['image/object/class/text'], default_value='')
#    labels = tf.cast(class_table.lookup(class_text), tf.float32)
#    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
#                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
#                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
#                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
#                        labels], axis=1)
#
#    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
#    y_train = tf.pad(y_train, paddings)
#
#    return x_train, y_train


#def load_tfrecord_dataset(file_pattern, class_file, size=416):
#    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
#    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
#        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)
#
#    files = tf.data.Dataset.list_files(file_pattern)
#    dataset = files.flat_map(tf.data.TFRecordDataset)
#    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


#def load_fake_dataset():
#    x_train = tf.image.decode_jpeg(
#        open('./data/girl.png', 'rb').read(), channels=3)
#    x_train = tf.expand_dims(x_train, axis=0)
#
#    labels = [
#        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
#        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
#        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
#    ] + [[0, 0, 0, 0, 0]] * 5
#    y_train = tf.convert_to_tensor(labels, tf.float32)
#    y_train = tf.expand_dims(y_train, axis=0)
#
#    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
