import tensorflow as tf


def conv2d(inputs, kernel_size, stride, depth, padding='SAME', subscope='0'):
    with tf.name_scope("conv2d_{}".format(subscope)):
        weights = tf.get_variable("weights_{}".format(subscope),
                                  shape=[kernel_size, kernel_size, inputs.get_shape()[3], depth],
                                  initializer=tf.contrib.layers.xavier_initializer()
                                  )
        biases = tf.get_variable("bias{}".format(subscope),
                                 shape=[depth],
                                 initializer=tf.constant_initializer(0.0)
                                 )
        # batch normalization for the input data
        _bn = tf.nn.batch_normalization(
            inputs,
            mean=0.0,
            variance=1.0,
            offset=1.0,
            scale=1.0,
            variance_epsilon=1.0,
            name="bn"
        )
        _relu = tf.nn.relu(_bn, name="relu_{}".format(subscope))

        # Convolution process for the activated data
        _conv = tf.nn.conv2d(_relu, weights, strides=[1, stride, stride, 1], padding=padding, name="conv_{}".format(subscope))
        
        # add bias for the data
        _output = tf.nn.bias_add(_conv, biases)
        print('{}| {} ---> {}'.format("conv2d_{}".format(subscope), inputs.get_shape(), _output.get_shape()))
        return _output


def deconv2d(inputs, factor=2, subscope='0'):
    with tf.name_scope("deconv2d_{}".format(subscope)):
        b, h, w, c = inputs.get_shape().as_list()
        n_h, n_w = h * factor, w * factor

        if h == 4:
            n_h = 7
            n_w = 7
        output = tf.image.resize_images(images=inputs, size=(n_h, n_w))
        print('{}| {} ---> {}'.format("deconv2d_{}".format(subscope), inputs.get_shape(), output.get_shape()))
        return output


def concat(x_1, x_2):
    with tf.name_scope("concat"):
        concated = tf.concat([x_1, x_2], axis=-1)
        print("Concat {} + {} --> {}".format(x_1.get_shape(), x_2.get_shape(), concated.get_shape()))
        return concated


def unet_module(inputs, subscope='1'):
    print("  unet-------")
    with tf.name_scope("encoder_{}".format(subscope)):
        conv1 = conv2d(inputs, kernel_size=1, stride=1, depth=64, subscope='{}_1'.format(subscope))
        conv2 = conv2d(conv1, kernel_size=3, stride=2, depth=32, subscope='{}_2'.format(subscope))
        conv3 = conv2d(conv2, kernel_size=3, stride=1, depth=32, subscope='{}_3'.format(subscope))
        conv4 = conv2d(conv3, kernel_size=3, stride=2, depth=16, subscope='{}_4'.format(subscope))
        conv5 = conv2d(conv4, kernel_size=3, stride=1, depth=16, subscope='{}_5'.format(subscope))

    with tf.name_scope("decoder_{}".format(subscope)):
        deconv1 = deconv2d(conv5, subscope='{}_1'.format(subscope))
        conv6 = conv2d(deconv1, kernel_size=11, stride=1, depth=32, subscope='{}_6'.format(subscope))
        concated = concat(conv3, conv6)
        deconv2 = deconv2d(concated, subscope='{}_2'.format(subscope))
        conv7 = conv2d(deconv2, kernel_size=3, stride=1, depth=64, subscope='{}_7'.format(subscope))
        conv8 = conv2d(conv7, kernel_size=1, stride=1, depth=64, subscope='{}_8'.format(subscope))

    with tf.name_scope("output_{}".format(subscope)):
        output = tf.identity(conv8 + inputs)

    return output