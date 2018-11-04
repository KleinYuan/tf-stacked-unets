# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:11:27 2018

@author: PeterLin
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf

def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)



#def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
#    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

#def deconv2d(x, W,stride):
#    with tf.name_scope("deconv2d"):
#        x_shape = tf.shape(x)
#        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
#        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name="conv2d_transpose")

#
#def crop_and_concat(x1,x2):
#    with tf.name_scope("crop_and_concat"):
#        x1_shape = tf.shape(x1)
#        x2_shape = tf.shape(x2)
#        # offsets for the top left corner of the crop
#        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
#        size = [-1, x2_shape[1], x2_shape[2], -1]
#        x1_crop = tf.slice(x1, offsets, size)
#        return tf.concat([x1_crop, x2], 3)




#def _variable_with_weight_decay(name, shape, stddev, wd):
#  """Helper to create an initialized Variable with weight decay.
#  Note that the Variable is initialized with a truncated normal distribution.
#  A weight decay is added only if one is specified.
#  Args:
#    name: name of the variable
#    shape: list of ints
#    stddev: standard deviation of a truncated Gaussian
#    wd: add L2Loss weight decay multiplied by this float. If None, weight
#        decay is not added for this Variable.
#  Returns:
#    Variable Tensor
#  """
# dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
#
# var = _variable_on_cpu(
#      name,
#      shape,
#      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
#  if wd is not None:
#    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#    tf.add_to_collection('losses', weight_decay)
#  return var