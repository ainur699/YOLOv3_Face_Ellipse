from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf

import matplotlib.pyplot as plt
import cv2

import numpy as np
import os
import random


def LoadFDDB(label_file, images_rot):
    data = []

    with open(label_file) as f:
        raw_label = f.read().strip().split('\n')
        
        id = 0
        
        while id < len(raw_label):
            image_name = os.path.join(images_rot, raw_label[id])
            image_name = os.path.normpath(image_name)
            ell_num = int(raw_label[id + 1])
            id += 2
            
            ellipses = []
            for i in range(ell_num):
                ell = tuple(map(float, raw_label[id+i].split(' ')))
                ell_norm = (ell[2], ell[3], ell[4], ell[0], ell[1])
                ellipses.append(ell_norm)
            
            id += ell_num
            
            data.append((image_name, ellipses))

    random.seed(1)
    random.shuffle(data)

    return data


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255.0

    shape = tf.shape(image)
    max_shape = tf.maximum(shape[0], shape[1])
    t_pad = (max_shape - shape[0]) // 2
    b_pad = max_shape - shape[0] - t_pad 
    l_pad = (max_shape - shape[1]) // 2
    r_pad = max_shape - shape[1] - l_pad

    image = tf.pad(image, [[t_pad, b_pad], [l_pad, r_pad], [0, 0]])
    image = tf.image.resize(image, (FLAGS.size, FLAGS.size))

    return image, (l_pad, t_pad), max_shape


def preprocess_data(x, y):
    # image
    image, pad, max_shape = load_and_preprocess_image(x)

    # label
    shift = [0, 0, 0, pad[0], pad[1]]
    y = tf.add(y, shift)

    ratio = 1.0 / tf.cast(max_shape, tf.float32);
    ratio = [ratio, ratio, 1, ratio, ratio]
    ratio = tf.cast(ratio, tf.float32)
    y = tf.multiply(y, ratio)

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y)[0]], [0, 0]]
    y = tf.pad(y, paddings)

    return (image, y)


def crop_rotate_flip(img, ellipses, scale_factor, rot_factor, flip):
    scale = random.uniform(1 - scale_factor, 1 + scale_factor)
    r     = random.uniform(-rot_factor, rot_factor) if random.random() <= 0.6 else 0
    
    if flip and random.random() <= 0.5:
        img = np.fliplr(img)
        for ell in ellipses:
            ell[3] = 1 - ell[3]
            ell[2] = -ell[2]
    
    trf = cv2.getRotationMatrix2D((FLAGS.size / 2, FLAGS.size / 2), r, scale)
    img = cv2.warpAffine(img, trf, (FLAGS.size, FLAGS.size))

    for ell in ellipses:
        if ell[0] == 0:
            break
        
        center = (int(FLAGS.size * ell[3]), int(FLAGS.size * ell[4]))
        center = np.array([[center]])
        center = cv2.transform(center, trf)
        
        ell[0] = scale * ell[0]
        ell[1] = scale * ell[1]
        ell[2] = ell[2] - np.pi / 180.0 * r
        ell[3] = center[0][0][0] / FLAGS.size
        ell[4] = center[0][0][1] / FLAGS.size

        if ell[3] < 0 or ell[3] > 1 or ell[4] < 0 or ell[4] > 1:
            ell[0] = ell[1] = 0

    return img, ellipses


def CreateFDDB(datasets, is_train, scale_factor=0.25, rot_factor = 30.0, flip=True):
    data = []
    for data_path in datasets:
        d = LoadFDDB(data_path[0], data_path[1])
        data = data + d

    data = np.array(data)
    data_x = data[:, 0]
    data_y = data[:, 1]

    x = tf.data.Dataset.from_tensor_slices(data_x)
    y = tf.data.Dataset.from_generator(lambda: data_y, tf.float32, [None, 5])
   
    dataset = tf.data.Dataset.zip((x, y))
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_train:
        numpy_fun = lambda x, y : crop_rotate_flip(x, y, scale_factor, rot_factor, flip)

        def tf_func(x, y):
            return tf.numpy_function(numpy_fun, [x, y], (tf.float32, tf.float32))

        dataset = dataset.map(tf_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


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
                box_xy = y_true[i][j][3:5]

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                if tf.equal(grid_xy[0], grid_size): 
                    grid_xy = tf.subtract(grid_xy, [1, 0])
                if tf.equal(grid_xy[1], grid_size):
                    grid_xy = tf.subtract(grid_xy, [0, 1])

                # grid[y][x][anchor] = (tx, ty, bw, bh, angle, obj)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [ell[3], ell[4], ell[0], ell[1], ell[2], 1])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack()) if idx > 0 else y_true_out


def DrawExample(example, name):
    image = 255.0 * example[0].numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for ell in example[1].numpy():
        center_coordinates = (int(FLAGS.size * ell[3]), int(FLAGS.size * ell[4]))
        axesLength = (int(FLAGS.size * ell[0]), int(FLAGS.size * ell[1]))
        angle = int(180.0 / 3.1416 * ell[2])

        cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, (0,255,0), 1)
 
    cv2.imwrite(name, image)


def DrawOutputs(img, outputs, name):
    im = 255 * img.numpy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


    ellipses = outputs.numpy()

    if True:
        for ell in ellipses:
            if ell[2] == 0:
                continue
            xywh = FLAGS.size * ell[0:4]
        
            center_coordinates = (int(xywh[0]), int(xywh[1]))
            axesLength = (int(xywh[2]), int(xywh[3]))
            angle = int(180.0 / 3.1416 * ell[4])
        
            cv2.ellipse(im, center_coordinates, axesLength, angle, 0, 360, (0,255,0), 1)
        
        cv2.imwrite(name, im)
    else:
        if len(ellipses) != 0 and ellipses[0][2] != 0:
            ell = ellipses[0]

            xywh = FLAGS.size * ell[0:4]

            center = np.array([xywh[0], xywh[1]])
            axes = np.array([xywh[2], xywh[3]])
            angle = ell[4]

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

            src = np.float32([bbox_tl, bbox_tr, bbox_br])
            dst = np.float32([[0,0], [511, 0], [511, 511]])

            trf = cv2.getAffineTransform(src, dst)
            im = cv2.warpAffine(im, trf, (512, 512))

            cv2.imwrite(name, im)


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
