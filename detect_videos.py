import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from models import YoloV3FaceTiny
import dataset as dat2
import os
from tqdm import tqdm
from glob import glob
import json

from voxtrain_dataset import VideoDataset
from torch.utils.data import DataLoader


flags.DEFINE_string('root_dir', 'D:/Github/video-preprocessing/vox1-png', 'root directory')
flags.DEFINE_string('weights', './checkpoints/model_best.tf', 'path to weights file')
flags.DEFINE_integer('size', 512, 'resize images to')


def get_bbox(ellipses):
    if ellipses[0][2] == 0:
        return None

    ell = ellipses[0]

    xy = 512 * ell[0:2]
    wh = 512 * ell[2:4]

    center = np.array([xy[0], xy[1]])
    axes = np.array([wh[0], wh[1]])
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
    
    bbox = {}
    bbox['tl'] = bbox_tl.tolist()
    bbox['tr'] = bbox_tr.tolist()
    bbox['br'] = bbox_br.tolist()

    return bbox


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3FaceTiny(FLAGS.size)
    yolo.load_weights(FLAGS.weights).expect_partial()

    root = 'C:/Github/video-preprocessing/vox-video/train'
    videos = glob(os.path.join(root, '*.mp4'))
    
    for video_path in tqdm(videos):
        dataset = VideoDataset(video_path)
        dataloader = DataLoader(dataset, batch_size=40, num_workers=0)

        bboxex = {}
        for data in dataloader:
            idx = data['id'].numpy()
            imgs = data['frame'].numpy()
            imgs = tf.convert_to_tensor(imgs, dtype=tf.float32).gpu()

            outputs = yolo(imgs)
            ellipses = outputs.numpy()

            for e, i in zip(ellipses, idx):
                bboxex[int(i)] = get_bbox(e)

        save_path = video_path.replace('train', 'bbox').replace('.mp4', '.json')
        with open(save_path, 'w') as f:
            json.dump(bboxex, f)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
