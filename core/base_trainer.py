import tensorflow as tf
import numpy as np
import random
import cv2

class BaseTrainer(object):
    session = None
    writer = None
    saver = None
    last_save_name = None
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    optimizer = None
    summary_op = None

    def __init__(self, model, data_model, config, logger):
        self.model = model
        self.epochs = config.train.epochs
        self.batch_size = config.train.batch_size
        self.logdir = config.file.logdir
        self.save_path = config.file.save_path
        self.val_epoch = config.train.val_epoch
        self.save_epoch = config.train.save_epoch
        self.config = config
        self.logger = logger
        self.data_model = data_model

    def _random_flip(self, x_batch):
        _augmented_x_batch = np.zeros_like(x_batch)
        should_flip = random.random() < 0.5
        for idx, _ in enumerate(x_batch):
            _augmented_x = x_batch[idx]
            if should_flip:
                flip_direction = random.choice([-1, 0, 1])
                _augmented_x = cv2.flip(_augmented_x, flip_direction)
            _augmented_x_batch[idx] = _augmented_x
        return _augmented_x_batch

    def _random_rotate(self, x_batch):
        _augmented_x_batch = np.zeros_like(x_batch)
        should_rotate = random.random() < 0.9
        for idx, _ in enumerate(x_batch):
            _augmented_x = x_batch[idx]
            if should_rotate:
                rotate_angle = int(random.random() * 360)
                _augmented_x = self._rotate(_augmented_x, rotate_angle)
            _augmented_x_batch[idx] = _augmented_x
        return _augmented_x_batch

    @staticmethod
    def _rotate(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0,)
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (w, h))

    def _epoch_train(self, epoch):
        x = self.x_train
        y = self.y_train

        train_loss_agg = 0
        _iter = 0
        for x_batch, y_batch in self.data_model.batch_iterator(x, y, batch_size=self.batch_size):
            _iter += 1
            x_batch = self._random_flip(x_batch)
            x_batch = self._random_rotate(x_batch)
            feed_dict = {
                self.model.x_pl: x_batch,
                self.model.y_pl: y_batch,
            }
            _, _train_loss, summary_train = self.session.run([self.model.optimizer, self.model.loss, self.summary_op], feed_dict=feed_dict)
            train_loss_agg += _train_loss
            pred = self.session.run([self.model.prediction], feed_dict=feed_dict)
            # print("pred: {}/{}/{}".format(pred[0][0], pred[0][0].shape, np.argmax(pred[0][0])))
            # print("gt: {}".format(y_batch[0]))
            self.logger.info('  {} th iter: train loss: {}'.format(_iter, _train_loss))
            self.writer_train.add_summary(summary_train, epoch)
        avg_train_loss = train_loss_agg / _iter
        self.logger.info('{} th epoch:  train loss: {}'.format(epoch, avg_train_loss))

        val_loss_agg = 0
        _val_iter = 0
        if (epoch % self.val_epoch) == 0:
            for x_batch, y_batch in self.data_model.batch_iterator(self.x_val, self.y_val, batch_size=self.batch_size):
                _val_iter += 1
                feed_dict = {
                    self.model.x_pl: x_batch,
                    self.model.y_pl: y_batch,
                }
                _val_loss, summary_val = self.session.run([self.model.loss, self.summary_op], feed_dict=feed_dict)
                val_loss_agg += _val_loss
                self.logger.info('{} th epoch:  val loss: {}'.format(epoch, _val_loss))
                self.writer_val.add_summary(summary_val, epoch)
            avg_val_loss = val_loss_agg / _val_iter
            self.logger.info('{} th epoch:  val loss: {}'.format(epoch, avg_val_loss))

        if (epoch % self.save_epoch) == 0 or (epoch == self.epochs - 1):
            snapshot_path = self.saver.save(sess=self.session, save_path="%s_%s" % (self.save_path, epoch))
            self.logger.info('Snapshot of {} th epoch is saved to {}'.format(epoch, snapshot_path))

    def train(self):
        self.logger.info('Start training ...')

        with tf.Session(graph=self.model.graph) as self.session:
            self.summary_op = tf.summary.merge_all()
            self.x_train, self.y_train = self.data_model.get_train()
            self.x_val, self.y_val = self.data_model.get_val()
            self.session.run(self.model.init_graph)
            self.saver = tf.train.Saver()
            self.writer_train = tf.summary.FileWriter(logdir=self.logdir + 'train/', graph=self.session.graph)
            self.writer_val = tf.summary.FileWriter(logdir=self.logdir + 'val/', graph=self.session.graph)

            for _epoch in range(0, self.epochs):
                self._epoch_train(epoch=_epoch)

            save_path = self.saver.save(self.session, self.save_path)
            self.logger.info('Training ended and model file is in here: ', save_path)
