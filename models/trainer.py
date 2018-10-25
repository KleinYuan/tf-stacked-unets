class Trainer(object):

	def __init__(self, model, config, logger):
		self.model = model
		self.config = config
		self.logger = logger

	def train(self):
		raise NotImplementedError