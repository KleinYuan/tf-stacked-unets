#from ..core.base_model import BaseModel
#from ..models.unet import Model as UnetModule
import tensorflow as tf
from utils import *

class Model(object):
     def __init__(self):
         self.define_loss()
         self.define_optimizer()
         self.forward()
        
     def define_loss(self):
         self.sunet_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)

     def define_optimizer(self):
          starter_learning_rate = 0.0002
          decay_steps = 1000
          global_step = 100000
          learning_rate = tf.train.cosine_decay(starter_learning_rate, global_step, decay_steps)
          self.sunet_train_op = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(self.sunet_loss)
        
     def forward(self):  
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.Y = tf.placeholder(tf.float32, [None, 10])        
        
        with tf.variable_scope("sunet"):
            W_conv0 = weight_variable([7, 7, 3, 64])
            b_conv0 = weight_variable([64])
            h_conv0 = tf.nn.conv2d(inputs, W_conv0, strides=[1, 2, 2, 1], padding='SAME')
            h_conv0 = tf.nn.relu(h_conv0 + b_conv0)
            
            W_conv1 = weight_variable([3, 3, 64, 128])
            b_conv1 = weight_variable([128])
            h_conv1 = tf.nn.conv2d(h_conv0, W_conv1, strides=[1, 2, 2, 1], padding='SAME')
            h_conv1 = tf.nn.relu(h_conv1, b_conv1)
                
            W_conv2 = weight_variable([3, 3, 128, 128])
            b_conv2 = weight_variable([128])
            h_conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            h_conv2 = tf.nn.relu(h_conv2, b_conv2)     
            combined = h_conv1 + h_conv2
        
            h_unet1 = Unet(combined)
# todo change the Unet to He's code
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

if __name__ == "__main__":
    X = tf.random_normal(shape = [10, 224, 224, 3])
    Y = tf.random_normal(shape = [10, 10])
    model = Model
    with tf.Session() as sess:
        sess.run


