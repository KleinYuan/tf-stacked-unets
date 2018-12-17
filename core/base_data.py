import numpy as np


class Model(object):
    data_sets = None
    x_train = None
    y_train = None
    x_val = None
    y_val = None

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._load_data()

    def _load_data(self):
        _data_name = self.config.name
        _num_data = self.config.number
        self.logger.info("Loading {}...".format(_data_name))
        if _data_name == 'mscoco':
            from tensor2tensor.data_generators import mscoco as _data_sets
            self.data_sets = _data_sets.mscoco_generator(
                data_dir='data/',
                tmp_dir='tmp/',
                training=True,
                how_many=_num_data
            )
            self.logger.info("Loaded {}...".format(_data_name))
        elif _data_name == 'cifar10':
            import tensorflow as tf
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.cifar10.load_data()
            # self.y_train = np.eye(10)[self.y_train[:, ]]
            # self.y_train = np.reshape(self.y_train, (self.y_train.shape[0], self.y_train.shape[-1]))
            # self.y_val = np.eye(10)[self.y_val[:, ]]
            # self.y_val = np.reshape(self.y_val, (self.y_val.shape[0], self.y_val.shape[-1]))

    @staticmethod
    def batch_iterator(xs, ys, batch_size=1):
        l = len(xs)
        assert l == len(ys)
        for ndx in range(0, l, batch_size):
            yield xs[ndx:min(ndx + batch_size, l)], ys[ndx:min(ndx + batch_size, l)]

    def get_train(self):
        return self.x_train, self.y_train

    def get_val(self):
        return self.x_val, self.y_val


def test():
    class DummyConfig:
        name = 'mscoco'
        number = 20

    class DummyLogger:
        def info(self, msg):
            print("[Info] {}".format(msg))

    config = DummyConfig()
    logger = DummyLogger()
    model = Model(config=config, logger=logger)
    assert model.data_sets is not None

    for _data in model.data_sets:
        print(type(_data))


if __name__ == "__main__":
    test()
