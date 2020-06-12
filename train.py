from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
import datetime

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from models import (
    YoloV3, YoloV3Tiny,
    YoloV3Face, YoloV3FaceTiny, YoloLoss,
    yolo_face_anchors, yolo_face_anchor_masks,
    yolo_face_tiny_anchors, yolo_face_tiny_anchor_masks
)
from utils import freeze_all
import dataset

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')


flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 
                    'path to weights file') # './checkpoints/yolov3-tiny.tf'
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'darknet',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune', 'resume'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only, '
                  'resume: Continue from the last point')
flags.DEFINE_integer('size', 512, 'image size')
flags.DEFINE_integer('epochs', 500, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Model
    if FLAGS.tiny:
        model = YoloV3FaceTiny(FLAGS.size, training=True)
        anchors = yolo_face_tiny_anchors
        anchor_masks = yolo_face_tiny_anchor_masks
    else:
        model = YoloV3Face(FLAGS.size, training=True)
        anchors = yolo_face_anchors
        anchor_masks = yolo_face_anchor_masks

  
    # Dataset
    train_dataset = dataset.CreateFDDB([('dataset/fddb/FDDB_train.txt', 'dataset/fddb/images'), ('dataset/300w/300w_train.txt', 'dataset/300w/images')], True) 
    val_dataset   = dataset.CreateFDDB([('dataset/300w/300w_valid.txt','dataset/300w/images')], False)

    #for i, data in enumerate(train_dataset.take(10)):
    #    dataset.DrawExample(data, 'log_' + str(i) + '.png')

    train_dataset = train_dataset.shuffle(512)
    train_dataset = train_dataset.batch(8)
    train_dataset = train_dataset.map(lambda x, y: (x, dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(8)
    val_dataset = val_dataset.map(lambda x, y: (x, dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))


    # transfer learning
    if FLAGS.transfer == 'none':
        pass
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(FLAGS.size, training=True)
        else:
            model_pretrained = YoloV3(FLAGS.size, training=True)

        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)
    elif FLAGS.transfer in ['fine_tune', 'frozen']:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen': 
            # freeze everything
            freeze_all(model)
    else:
        # resume
        model.load_weights('checkpoints/model_best.tf')


    lr = 1e-3
    step_patient = 0
    def lr_schedule():
        return lr

    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    loss = [YoloLoss(anchors[mask]) for mask in anchor_masks]
    best_valid_loss =  float('inf')

    logdir = "logs/" + datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()
    loss_step = 0
    loss_valid_step = 0


    if FLAGS.mode == 'eager_tf':
        avg_loss     = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                avg_loss.update_state(total_loss)
                if batch % 20 == 0:
                    logging.info("{}_train_{}, {}, {}".format(epoch, batch, total_loss.numpy(),list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    tf.summary.scalar('loss', data=total_loss, step=loss_step)
                    tf.summary.flush()
                    loss_step = loss_step + 1

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                avg_val_loss.update_state(total_loss)

                if batch % 20 == 0:
                    logging.info("{}_val_{}, {}, {}".format(epoch, batch, total_loss.numpy(), list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                

            is_best = avg_val_loss.result().numpy() < best_valid_loss
            best_valid_loss = min(avg_val_loss.result(), best_valid_loss)

            logging.info('cur step: {}'.format(optimizer.iterations.numpy()))
            logging.info('cur base learning rate: {}'.format(optimizer.lr))

            logging.info("best: {}".format(is_best))
            logging.info("{}, train: {}, val: {}".format(epoch, avg_loss.result().numpy(), avg_val_loss.result().numpy()))
            tf.summary.scalar('loss_valid', data=avg_val_loss.result(), step=loss_valid_step)
            tf.summary.flush()
            loss_valid_step = loss_valid_step + 1

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights('checkpoints/train_{}.tf'.format(epoch))
            model.save_weights('checkpoints/latest.tf')

            if is_best:
                model.save_weights('checkpoints/model_best.tf')
                step_patient = 0
            else:
                step_patient = step_patient + 1
                logging.info('patient step: {}'.format(step_patient))

            if step_patient >= 4:
                lr = 0.1 * lr
                step_patient = 0
                logging.info('reduce learning rate until {}'.format(lr))

            if step_patient > 15:
                logging.info('exit learning loop by stopping')
                break



    else:
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(factor=0.3, patience=6, verbose=1),
            EarlyStopping(patience=15, verbose=1),
            ModelCheckpoint('checkpoints/train_{epoch}.tf', verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
