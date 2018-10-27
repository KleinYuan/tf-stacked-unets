class BaseTrainer(object):

    session = None
    writer = None
    last_save_name = None

    def __init__(self, model, config, logger):
        self.model = model
        self.epochs = config.train.epochs
        self.batch_size = config.train.batch_size
        self.logdir = config.file.logdir
        self.save_path = config.file.save_path
        self.val_epoch = config.train.val_epoch
        self.save_epoch = config.train.save_epoch
        self.config = config
        self.logger = logger
        self.init()

    def init(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
