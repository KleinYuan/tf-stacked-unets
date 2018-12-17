import tensorflow as tf


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

    def _epoch_train(self, epoch):
        x = self.x_train
        y = self.y_train

        train_loss_agg = 0
        _iter = 0
        for x_batch, y_batch in self.data_model.batch_iterator(x, y, batch_size=self.batch_size):
            _iter += 1
            feed_dict = {
                self.model.x_pl: x_batch,
                self.model.y_pl: y_batch,
            }
            _, _train_loss, summary = self.session.run([self.model.optimizer, self.model.loss, self.summary_op], feed_dict=feed_dict)
            train_loss_agg += _train_loss
            self.logger.info('  {} th iter: train loss: {}'.format(_iter, _train_loss))
            self.writer.add_summary(summary, epoch)
        avg_train_loss = train_loss_agg / _iter
        self.logger.info('{} th epoch:  train loss: {}'.format(epoch, avg_train_loss))

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
            self.writer = tf.summary.FileWriter(logdir=self.logdir, graph=self.session.graph)

            for _epoch in range(0, self.epochs):
                self._epoch_train(epoch=_epoch)

            save_path = self.saver.save(self.session, self.save_path)
            self.logger.info('Training ended and model file is in here: ', save_path)
