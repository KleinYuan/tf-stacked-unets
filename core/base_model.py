import tensorflow as tf


class Model(object):
    loss = None
    optimizer = None
    x_pl = None
    y_pl = None
    prediction = None
    graph = None
    summary_list = None
    init_graph = None
    learning_rate = None

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model_name = config.name

        self.define_graph()

    def define_summary_list(self):
        self.summary_list = [
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("lr", self.learning_rate)
        ]

    def define_loss(self):
        raise NotImplementedError

    def define_optimizer(self):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError

    def define_graph(self):
        self.logger.info('Constructing graph now...')
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.define_net()
            self.define_loss()
            self.define_optimizer()
            self.define_summary_list()
            self.init_graph = tf.global_variables_initializer()

        self.logger.info('Graph constructed!')

    def define_net(self):
        with tf.name_scope(self.model_name):
            self.x_pl = tf.placeholder(
                dtype=tf.float32,
                shape=[None] + self.config.inputs.x_pl.shape,
                name=self.config.inputs.x_pl.name
            )
            self.y_pl = tf.placeholder(
                dtype=tf.int32,
                shape=[None] + self.config.inputs.y_pl.shape,
                name=self.config.inputs.y_pl.name
            )
            self.prediction = self.forward(inputs=self.x_pl)

    def get_train_ops(self):
        return [self.init_graph, self.graph]
