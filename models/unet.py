# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:39:48 2018

@author: hezhang
"""

import tensorflow as tf
from layer import (conv2d, deconv2d, concat, weight_variable, bias_variable)
import logging
import numpy as np

def _down_stream(inputs, stride, keep_prob, weight_shape, bias_shape, scope, logger):     
    with tf.name_scope(scope):
        weights = weight_variable(weight_shape, name = "weights")
        bias = bias_variable(bias_shape, name="bias")
        output = conv2d(inputs, weights, stride, bias, keep_prob)
        logger.info("{} ---> {}".format(inputs, output))
        return output, weights, bias


def _up_stream(inputs, stride, weight_shape, scope, logger):
    with tf.name_scope(scope):
        weights = weight_variable(weight_shape, name = "w_up_1")
        output = deconv2d(inputs, weights, stride)
        logger.info("{} ---> {}".format(inputs, output))
        return output, weights



def unet_module(x_unet, input_channels, logger, keep_prob, feature_map=64, filter_size_bound=1, filter_size=3):
    """
        Create a new convolutional unet with the given paramerters
        :param x:                  image input
        :param batch_size:         the size of the batch at each training step
        :param input_channels:     the number of channels of the input image
        :param feature_map:        the feature_map of each filter, corresponds to N in the original paper
        :param filter_size_bound:  the filter size for the first and last convolutional layers
        :param filter_size         the filter size the the inside convolutional layers
    """

    # placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x_unet)[1]
        ny = tf.shape(x_unet)[2]
        x_image = tf.reshape(x_unet, tf.stack([-1, nx, ny, input_channels]))
        x_image = tf.cast(x_image, tf.float32)
        in_node = x_image


    # specify the weight_shape for each layer
    weight_shape_input = [filter_size_bound, filter_size_bound, input_channels, feature_map]
    weight_shape_output = [filter_size, filter_size, feature_map, input_channels]
    weight_shape_inner = [filter_size, filter_size, feature_map, feature_map]
    weight_shape_concat = [filter_size, filter_size, feature_map, input_channels]
    b_shape = [feature_map]
    keep_p = keep_prob
    
    """
        According to the paper of the stacked u-net, our code is designed for the u-net model in that paper specifically, 
        in the future version, we will design a general u-net framework for the stacked u-net
        1 down layer:   a 1*1 convolutional layer with stride = 1, the output is a N*64*64 image
        2 down layer:   a 3*3 convolutional layer with stride = 2, the output is a N*32*32 image
        3 down layer:   a 3*3 convolutional layer with stride = 1, the output is a N*32*32 image
        4 down layer:   a 3*3 convolutional layer with stride = 2, the output is a N*16*16 image
        5 down layer:   a 3*3 convolutional layer with stride = 1, the output is a N*16*16 image

        1 up layer:           a 3*3 convolutional layer with stride = 2, the output is a N*32*32 image
        2 up layer:           a 3*3 convolutional layer with stride =1, the output is a N*32*32 image
        concatenation layer:  the output of the second up layer is concatenated with the output of the third layer 
        3 up layer:           a 3*3 convolutional layer with stride =2, the output is a N*64*64 image
        4 up layer:           a 3*3 convolutional layer with stride =1, the output is a N*64*64 image
        5 up layer:           a 3*3 convolutional layer with stride =1, and feature_map = input_channels, the output is a M*64*74 image

    """


    conv1, w_down_1, b1 = _down_stream(inputs=in_node, stride=1, keep_prob=keep_p, weight_shape=weight_shape_input,
                                       bias_shape=b_shape, logger=logger, scope="conv1")
    conv2, w_down_2, b2 = _down_stream(inputs=conv1, stride=2, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv2")
    conv3, w_down_3, b3 = _down_stream(inputs=conv2, stride=1, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv3")
    conv4, w_down_4, b4 = _down_stream(inputs=conv3, stride=2, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv4")
    conv5, w_down_5, b5 = _down_stream(inputs=conv4, stride=1, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv5")

    deconv1, w_up_1 = _up_stream(inputs=conv5, stride=2, weight_shape=weight_shape_inner,
                                 logger=logger, scope="deconv1")
    deconv2, w_up_2 = _up_stream(inputs=deconv1, stride=1, weight_shape=weight_shape_inner,
                                 logger=logger, scope="deconv2")
    concat_layer = concat(conv3, deconv2)
    deconv3, w_up_3 = _up_stream(inputs=concat_layer, stride=2, weight_shape=weight_shape_concat,
                                 logger=logger, scope="deconv3")
    deconv4, w_up_4 = _up_stream(inputs=deconv3, stride=1, weight_shape=weight_shape_inner,
                                 logger=logger, scope="deconv4")
    deconv5, w_up_5 = _up_stream(inputs=deconv4, stride=1, weight_shape=weight_shape_output,
                                 logger=logger, scope="deconv5")



    return deconv5



if __name__ == "__main__":
    test_logger = logging.getLogger('Mylogger')
    x_test = tf.placeholder(tf.float32, shape = [1,64,64,128])
    output = unet_module(x_unet = x_test, input_channels=128, logger=test_logger, keep_prob=0.5, feature_map=64, filter_size_bound=1, filter_size=3)
    with tf.Session() as sess:  
        rand_array = np.random.rand(1,64,64,128)
        print(sess.run(output, feed_dict={x_test: rand_array}))





        








