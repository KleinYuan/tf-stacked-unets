import sys
from ruamel.yaml import YAML
from box import Box
import tensorflow as tf
from models.sunet64_u1_model import Model
from core.base_trainer import BaseTrainer as Trainer
from core.base_data import Model as DataModel

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.logging


def run(config):
    data_model = DataModel(config=config.data, logger=logger)
    model = Model(config=config.model, logger=logger)
    trainer = Trainer(model=model, data_model=data_model, config=config, logger=logger)
    trainer.train()


if __name__ == "__main__":
    _config = Box(YAML(typ='safe').load(open(sys.argv[1]).read()))
    run(config=_config)
