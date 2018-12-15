from core.base_model import Model as BaseModel
#from ..models.unet import Model as UnetModule
import tensorflow as tf
from utils.layers import conv_relu


class Model(BaseModel):

     def define_loss(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y_pl)

     def define_optimizer(self):
        starter_learning_rate = 0.0002
        decay_steps = 1000
        global_step = 100000
        learning_rate = tf.train.cosine_decay(starter_learning_rate, global_step, decay_steps)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(self.loss)

     def forward(self, inputs):
        
        with tf.variable_scope("sunet"):
            conv1  = conv_relu(inputs, kernel_size=7, depth=64, stride=2, activatiton=True, name_scope="conv2d_1", padding='SAME')
            conv2 = conv_relu(conv1, kernel_size=3, depth=128, stride=2, activatiton=True, name_scope="conv2d_2", padding='SAME')
            conv3 = conv_relu(conv2, kernel_size=3, depth=128, stride=1, activatiton=True, name_scope="conv2d_3", padding='SAME')
            concat_1_2 = tf.concat([conv1, conv2], axis=-1, name='concat_1_2')

        
            h_unet1 = Unet(combined)
            h_pool1 = tf.nn.max_pool(h_unet1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
            h_unet2 = Unet(h_pool1)
            h_pool2 = tf.nn.max_pool(h_unet2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        
            
            h_unet3 = Unet(h_pool2)
            h_pool3 = tf.nn.max_pool(h_unet3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        
            h_unet4 = Unet(h_pool3)
            h_pool4 = tf.nn.avg_pool(h_unet4, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='SAME')
        
            W_fc0 = weight_variable([1024, 1000])
            b_fc0 = bias_variable([1000])
            logits = tf.matmul(h_pool4, W_fc0) + b_fc0
            pred = tf.nn.softmax(logits)
            return pred

