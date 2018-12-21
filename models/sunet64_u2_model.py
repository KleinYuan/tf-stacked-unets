"""
@author: kaiwen
"""

from core.base_model import Model as BaseModel
import tensorflow as tf
from utils.blocks import conv2d, unet_module


class Model(BaseModel):

     def define_loss(self):
        cross_entropy_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=self.prediction, labels=self.y_pl))
        vars   = tf.trainable_variables() 
        l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.00005
        self.loss = cross_entropy_loss + l2_loss
        tf.summary.scalar("l2-loss", l2_loss)
        tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

     def define_optimizer(self):
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.starter_lr, global_step, 10000, 0.96, staircase=True)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True).minimize(self.loss, global_step=global_step)

     def forward(self, inputs):

        with tf.name_scope("conv_layer"):
            print("conv_layer--------------")
            conv1 = conv2d(inputs, kernel_size=7, stride=2, depth=64, subscope='s1')
            
        
        with tf.name_scope("residual_block_layer"):
            print("residual_block_layer--------------")
            conv21 = conv2d(conv1, kernel_size=3, stride=2, depth=128, subscope='s21')
            conv22= conv2d(conv21, kernel_size=3, stride=1, depth=128, subscope='s22')

        with tf.name_scope("unet_block_1"):
            print("unet_block_1--------------")
            conv31 = conv2d(conv22, kernel_size=1, stride=1, depth=64, subscope='s31')
            conv32 = unet_module(inputs=conv31, subscope='s32')
            conv33 = conv2d(conv32, kernel_size=1, stride=1, depth=256, subscope='s33')

            conv34 = conv2d(conv33, kernel_size=1, stride=1, depth=64, subscope='s34')
            conv35 = unet_module(inputs=conv34, subscope='s35')
            conv36 = conv2d(conv35, kernel_size=1, stride=1, depth=256, subscope='s36')

        with tf.name_scope("transition_layer_1"):
            print("transition_layer_1--------------")
            pool41 = tf.nn.avg_pool(conv36, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

        with tf.name_scope("unet_block_2"):
            print("unet_block_2--------------")
            conv51 = conv2d(pool41, kernel_size=1, stride=1, depth=64, subscope='s51')
            conv52 = unet_module(inputs=conv51, subscope='s52')
            conv53 = conv2d(conv52, kernel_size=1, stride=1, depth=512, subscope='s53')

            conv54 = conv2d(conv53, kernel_size=1, stride=1, depth=64, subscope='s54')
            conv55 = unet_module(inputs=conv54, subscope='s55')
            conv56 = conv2d(conv55, kernel_size=1, stride=1, depth=512, subscope='s56')


        with tf.name_scope("classification_layer"):
            print("classification_layer--------------")
            pool100 = tf.reduce_mean(conv56, axis=[1, 2])
            print('{}| {} ---> {}'.format("global_pooling", conv56.get_shape(), pool100.get_shape()))
            output = tf.layers.dense(inputs=pool100, units=self.config.num_class, name='fc', activation=None)
            # output = tf.nn.softmax(fc, name='output')
        print("Output ------> {}".format(output.get_shape()))
        return output