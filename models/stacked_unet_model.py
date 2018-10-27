from ..core.base_model import BaseModel


class Model(BaseModel):

	def define_loss(self):
		raise NotImplementedError

	def define_optimizer(self):
		raise NotImplementedError

	def forward(self, inputs):
		raise NotImplementedError
