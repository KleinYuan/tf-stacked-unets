import sys
from ruamel.yaml import YAML
from box import Box
import tensorflow as tf
from ..models.stacked_unet_model import Model
from ..core.base_trainer import BaseTrainer as Trainer

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.logging


def run(config):
	model = Model(config=config.model, logger=logger)
	trainer = Trainer(model=model, config=config.train, logger=logger)

	trainer.train()


if __name__ == "__main__":
	_config = Box(YAML(typ='safe').load(open(sys.argv[1]).read()))
	run(config=_config)