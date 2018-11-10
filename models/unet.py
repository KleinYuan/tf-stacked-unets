﻿  # !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:39:48 2018

@author: hezhang
"""

import tensorflow as tf
from layer import (conv2d, deconv2d, concat, weight_variable, bias_variable, pixel_wise_softmax, cross_entropy)   


def _down_stream(inputs, stride, keep_prob, weight_shape, bias_shape, scope, logger):     
    with tf.name_scope(scope):
        weights = weight_variable(weight_shape, name = "weights")
        bias = bias_variable(bias_shape, name="bias")
        output = conv2d(inputs, weights, stride, bias, "NHWC", keep_prob)
        logger.info("{} ---> {}".format(inputs, output))
        return output, weights, bias


def _up_stream(inputs, stride, weight_shape, scope, logger):
    with tf.name_scope(scope):
        weights = weight_variable(weight_shape, name = "w_up_1")
        output = deconv2d(inputs, weights, stride, "NHWC")
        logger.info("{} ---> {}".format(inputs, output))
        return output, weights



def create_unet(x, input_channels, logger, keep_prob, batch_size=22, feature_map=64, filter_size_bound=1, filter_size=3,
                summaries=True):
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
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, input_channels]))
        in_node = x_image

# =============================================================================
#     # placeholder for the optimization variables and temprorary data
     variable_record = []
#     conv = []
#     deconv = OrderedDict()
#     dw_h_conv = OrderedDict()
#     up_h_convs = OrderedDict()
# 
#     in_size = 1000
#     size = in_size
# 
# =============================================================================

    # specify the weight_shape for each layer
    weight_shape_input = [filter_size_bound, filter_size_bound, input_channels, feature_map]
    weight_shape_output = [filter_size, filter_size, feature_map, input_channels]
    weight_shape_inner = [filter_size, filter_size, feature_map, feature_map]
    b_shape = [feature_map]
    keep_p = keep_prob
    
    ## -------- layers definition ------------
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
# =============================================================================
#     # ----- 1 down layer ------
#     with tf.name_scope("1_down_layer"):
#         w_down_1 = weight_variable(
#             [filter_size_bound, filter_size_bound, input_channels，feature_map], name = "w_down_1")
#         b_down_1 = bias_variable([feature_map], name="b_down_1")
#         conv1 = conv2d(in_node, w_down_1, stride=1, b_down_1, "NHWC", keep_prob)
#         weights.append((w_down_1))
#         biases.append((b_down_1))
#         conv.append((conv1))
# 
#     # ----- 2 down layer -----
#     with tf.name_scope("2_down_layer"):
#         w_down_2 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_down_2")
#         b_down_2 = bias_variable([feature_map], name="b_down_2")
#         conv2 = conv2d(conv1, w_down_2, stride=2, b_down_2, "NHWC", keep_prob)
#         weights.append((w_down_2))
#         biases.append((b_down_2))
#         conv.append((conv2))
# 
#     # ----- 3 down layer -----
#     with tf.name_scope("3_down_layer"):
#         w_down_3 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_down_3")
#         b_down_3 = bias_variable([feature_map], name="b_down_3")
#         conv3 = conv2d(conv2, w_down_3, stride=1, b_down_3, "NHWC", keep_prob)
#         weights.append((w_down_3))
#         biases.append((b_down_3))
#         conv.append((conv3))
# 
#     # ----- 4 down layer -----
#     with tf.name_scope("4_down_layer"):
#         w_down_4 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_down_4")
#         b_down_4 = bias_variable([feature_map], name="b_down_3")
#         conv4 = conv2d(conv3, w_down_3, stride=2, b_down_3, "NHWC", keep_prob)
#         weights.append((w_down_4))
#         biases.append((b_down_4))
#         conv.append((conv4))
# 
#     # ----- 5 down layer -----
#     with tf.name_scope("5_down_layer"):
#         w_down_5 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_down_5")
#         b_down_5 = bias_variable([feature_map], name="b_down_3")
#         conv5 = conv2d(conv4, w_down_4, stride=1, b_down_4, "NHWC", keep_prob)
#         weights.append((w_down_5))
#         biases.append((b_down_5))
#         conv.append((conv5))
# 
# =============================================================================


    # ----- down layers -----
    conv1, w_down_1, b1 = _down_stream(inputs=in_node, stride=1, keep_prob=keep_p, 
                                       weight_shape=weight_shape_input, bias_shape=b_shape, logger=logger, scope="conv1")
    conv2, w_down_2, b2 = _down_stream(inputs=conv1, stride=2, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv2")
    conv3, w_down_3, b3 = _down_stream(inputs=conv2, stride=1, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv3")
    conv4, w_down_4, b4 = _down_stream(inputs=conv3, stride=2, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv4")
    conv5, w_down_5, b5 = _down_stream(inputs=conv4, stride=1, keep_prob=keep_p, 
                                       weight_shape=weight_shape_inner, bias_shape=b_shape, logger=logger, scope="conv5")

    # ----- up layers -----
    deconv1, w_up_1 = _up_stream(inputs=conv5, stride=2, weight_shape=weight_shape_inner, logger=logger, scope="deconv1")
    deconv2, w_up_2 = _up_stream(inputs=deconv1, stride=1, weight_shape=weight_shape_inner, logger=logger, scope="deconv2")
    concat_layer = concat(conv3, deconv2)
    deconv3, w_up_3 = _up_stream(inputs=concat_layer, stride=2, weight_shape=weight_shape_inner, logger=logger, scope="deconv3")
    deconv4, w_up_4 = _up_stream(inputs=deconv3, stride=1, weight_shape=weight_shape_inner, logger=logger, scope="deconv4")
    deconv5, w_up_5 = _up_stream(inputs=deconv4, stride=1, weight_shape=weight_shape_output, logger=logger, scope="deconv5")

    variable_record.append((w_down_1,w_down_2,w_down_3,w_down_4,w_down_5,w_up_1,w_up_2,w_up_3,w_up_4,w_up_5,b1,b2,b3,b4,b5))



    if summaries:



    return deconv5, variable_record
# =============================================================================
#     # ----- 1 up layer -----
#     with tf.name_scope("1_up_layer"):
#         w_up_1 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_up_1")
#         deconv_1 = deconv2d(conv5, w_up_1, stride=2, "NHWC")
#         weights.append((w_up_1))
#         conv.append((deconv_1))
# 
#         # ----- 2 up layer -----
#     with tf.name_scope("2_up_layer"):
#         w_up_2 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_up_2")
#         deconv_2 = deconv2d(deconv_1, w_up_2, stride=1, "NHWC")
#         weights.append((w_up_2))
#         conv.append((deconv_2))
# 
#     # concatenation layer
#     with tf.name_scope("concatenation"):
#         concat_layer = concat(conv3, deconv_2)
# 
#     # ----- 3 up layer -----
#     with tf.name_scope("3_up_layer"):
#         w_up_3 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_up_3")
#         deconv_3 = deconv2d(concat_layer, w_up_3, stride=2, "NHWC")
#         weights.append((w_up_3))
#         conv.append((deconv_3))
# 
#     # ----- 4 up layer -----
#     with tf.name_scope("4_up_layer"):
#         w_up_4 = weight_variable([filter_size, filter_size, feature_map，feature_map], name = "w_up_4")
#         deconv_4 = deconv2d(deconv_3, w_up_4, stride=1, "NHWC")
#         weights.append((w_up_4))
#         conv.append((deconv_4))
# 
#     # ----- 5 up layer -----
#     with tf.name_scope("5_up_layer"):
#         w_up_5 = weight_variable([filter_size, filter_size, feature_map，input_channels], name = "w_up_5")
#         deconv_5 = deconv2d(deconv_4, w_up_5, stride=1, "NHWC")
#         weights.append((w_up_5))
#         conv.append((deconv_5))
# 
#     return deconv_5, weights, biases
# 
#     # visualize the u-net graph in tensorboard
# 
# =============================================================================

