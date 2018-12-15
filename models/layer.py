'''  
   this file includes basic functions to form a u-net:
       convolutional layer:     (input, filter, stride, bias, data_format, keep_probability) 
       deconvolutional layer:   (input, filter, stride, data_format)
       concatenation:  (input1, input2)
       
    some other functions: 
        
        
'''

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:12:53 2018

@author: hezhang
"""

import tensorflow as tf


# ---- convolutional layer----




# ----- a function to initailize the weight variables ------
def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

# =============================================================================
# # ----- a function to initailize the deconvolutional variables
# def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
#     return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
# 
# =============================================================================

# ----- a function to initailzie the bias variables
def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# ----- convolution layer
''' 
    x: input, 
    W: filter, 
    stride: stride size, 
    b: bias, 
    dformat: data format
    keep_prob_: keep probability
'''    
def conv2d(x_con, W, stride, b, keep_prob_):
    with tf.name_scope("conv2d"):
        # batch normalization for the input data
        x_norm = tf.nn.batch_normalization(x_con, mean=0.0, variance=1.0, offset=1.0 , scale=1.0, variance_epsilon=1.0, name="x_norm")
        
        # Relu activation for the normalized input
        x_act = tf.nn.relu(x_norm, name="x_act")
        
        # Convolution process for the activated data
        x_conv = tf.nn.conv2d(x_act, W, strides=[1, stride, stride, 1], padding='SAME', name="x_conv")
        
        # add bias for the data 
        x_conv_bias = tf.nn.bias_add(x_conv, b)
        return tf.nn.dropout(x_conv_bias, keep_prob_)
    
    
# ---- Deconvolutional layer -----
''' 
    x: input, 
    W: filter, 
    stride: stride size, 
    b: bias, 
    dformat: data format
'''
def deconv2d(x_dec, W, stride): 
    with tf.name_scope("deconv2d"):
        # obtain the shape of the input data x
        x_shape = tf.shape(x_dec)
        # obtain the output_shape
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]])
        return tf.nn.conv2d_transpose(x_dec, W, output_shape, strides=[1, stride, stride, 1], padding='SAME', name="conv2d_transpose")
        
    
    
# ---- concatenation layer
'''
    x_1 = first input
    x_2 = second_input
'''
def concat(x_1, x_2):
    with tf.name_scope("concat"):
        return tf.concat([x_1,x_2], axis = -1)



def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    


