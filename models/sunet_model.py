"""
@author: kaiwen
"""

from core.base_model import Model as BaseModel
import tensorflow as tf
from utils.blocks import conv_relu, unet_module


class Model(BaseModel):

     def define_loss(self):
        print(self.prediction)
        print(self.y_pl)
        self.loss = tf.losses.sparse_softmax_cross_entropy(logits=self.prediction, labels=self.y_pl)

     def define_optimizer(self):
        self.learning_rate = tf.train.cosine_decay(self.config.starter_lr, self.config.global_step, self.config.decay_steps)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.95).minimize(self.loss)

     def forward(self, inputs):

        with tf.name_scope("conv_layer"):
            conv1 = conv_relu(inputs, kernel_size=7, depth=64, stride=2, activatiton=True, name_scope="input_conv", padding='SAME')

        with tf.name_scope("residual_block_layer"):
            conv2 = conv_relu(conv1, kernel_size=3, depth=128, stride=2, activatiton=True, name_scope="conv2d_2", padding='SAME')
            conv3 = conv_relu(conv2, kernel_size=3, depth=128, stride=1, activatiton=True, name_scope="conv2d_3", padding='SAME')

        with tf.name_scope("unet_block_1"):
            conv4 = conv_relu(conv3, kernel_size=1, depth=64, stride=1, activatiton=True, name_scope="conv2d_4", padding='SAME')
            conv5 = unet_module(inputs=conv4, subscope='1')
            conv6 = conv_relu(conv5, kernel_size=1, depth=256, stride=1, activatiton=True, name_scope="conv2d_6", padding='SAME')

        with tf.name_scope("transition_layer_1"):
            pool7 = tf.nn.avg_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope("unet_block_2"):
            conv8 = conv_relu(pool7, kernel_size=1, depth=64, stride=1, activatiton=True, name_scope="conv2d_8", padding='SAME')
            conv9 = unet_module(inputs=conv8, subscope='2')
            conv10 = conv_relu(conv9, kernel_size=1, depth=512, stride=1, activatiton=True, name_scope="conv2d_10", padding='SAME')

        with tf.name_scope("transition_layer_2"):
            pool11 = tf.nn.avg_pool(conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope("unet_block_3"):
            conv12 = conv_relu(pool11, kernel_size=1, depth=64, stride=1, activatiton=True, name_scope="conv2d_12", padding='SAME')
            conv13 = unet_module(inputs=conv12, subscope='3')
            conv14 = conv_relu(conv13, kernel_size=1, depth=768, stride=1, activatiton=True, name_scope="conv2d_14", padding='SAME')

        with tf.name_scope("transition_layer_3"):
            pool15 = tf.nn.avg_pool(conv14, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope("unet_block_4"):
            conv16 = conv_relu(pool15, kernel_size=1, depth=64, stride=1, activatiton=True, name_scope="conv2d_16", padding='SAME')
            conv17 = unet_module(inputs=conv16, subscope='4')
            conv18 = conv_relu(conv17, kernel_size=1, depth=1024, stride=1, activatiton=True, name_scope="conv2d_18", padding='SAME')

        with tf.name_scope("classification_layer"):
            pool19 = tf.nn.avg_pool(conv18, ksize=[1, 7, 7, 1], strides=[1, 2, 2, 1], padding='SAME')
            output = tf.layers.dense(inputs=pool19, units=self.config.num_class, name='output')
            output = tf.reshape(output, (tf.shape(output)[0], self.config.num_class), name='output/reshaped')
        print("Output ------> {}".format(output.get_shape()))
        return output