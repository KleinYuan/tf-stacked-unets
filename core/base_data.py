import cv2
import os
import tarfile
import random


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
        self.logger.info("Loading {}...".format(_data_name))
        if _data_name == 'mscoco':
            from tensor2tensor.data_generators import mscoco as _data_sets
            _num_data = self.config.number
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
        elif _data_name == 'ILSVRC2012':
            _img_dir = self.config.img_dir
            _label_fp = self.config.label_fp
            _label_dict = {}
            _idx_ls = set()
            with open(_label_fp) as f:
                labels_content = f.readlines()
            for _cls in labels_content:
                _cls_split = _cls.split(' ')
                _label_dict[_cls_split[0]] = {
                    'idx': _cls_split[1],
                    'name': _cls_split[2]
                }

            print("Extracting data from .tar ......")
            _img_extracted_dir_ls = []
            for (dirpath, dirnames, filenames) in os.walk(_img_dir):
                for filename in filenames:
                    if filename.endswith('.tar'):
                        WNID = filename.split('.')[0]
                        _idx_ls.add(int(_label_dict[WNID]['idx']))
                        _abs_dir = os.sep.join([dirpath, filename])
                        _extracted_to = os.sep.join([dirpath, WNID]) + '/'
                        if not self.config.extracted:
                            print("  Extracting {} .tar ......".format(WNID))
                            _tar = tarfile.open(_abs_dir)
                            _tar.extractall(path=_extracted_to)
                            _tar.close()
                        _img_extracted_dir_ls.append(_extracted_to)
            print("Constructing training data sets .....")
            x_data = []
            y_data = []
            _idx_ls = sorted(list(_idx_ls))
            for _img_extracted_dir in _img_extracted_dir_ls:
                for (dirpath, dirnames, filenames) in os.walk(_img_extracted_dir):
                    for filename in filenames:
                        if filename.endswith('.JPEG'):
                            WNID = filename.split('.')[0].split('_')[0]
                            _abs_dir = os.sep.join([dirpath, filename])
                            _x = cv2.imread(_abs_dir)
                            _x = cv2.resize(_x, (224, 224))
                            x_data.append(_x)
                            _idx = int(_label_dict[WNID]['idx'])
                            _reranged_idx = _idx_ls.index(_idx)
                            y_data.append([_reranged_idx])

            shuffle_ls = list(zip(x_data, y_data))
            random.shuffle(shuffle_ls)
            x_data, y_data = zip(*shuffle_ls)
            split_idx = int(self.config.split_ratio * len(x_data))
            self.x_train = x_data[:split_idx]
            self.y_train = y_data[:split_idx]
            self.x_val = x_data[split_idx:]
            self.y_val = y_data[split_idx:]


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
