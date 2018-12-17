import tensorflow as tf


def conv_relu(input, kernel_size, depth, stride=1, activatiton=True, name_scope="conv2d", padding='SAME'):
    print('{} input =   {}'.format(name_scope, input.get_shape()))
    with tf.name_scope(name_scope):
        weights = tf.get_variable('{}/weights'.format(name_scope),
                                  shape=[kernel_size, kernel_size, input.get_shape()[3], depth],
                                  initializer=tf.contrib.layers.xavier_initializer()
                                  )
        biases = tf.get_variable('{}/biases'.format(name_scope),
                                 shape=[depth],
                                 initializer=tf.constant_initializer(0.0)
                                 )
        conv = tf.nn.conv2d(input=input,
                            filter=weights,
                            strides=[1, stride, stride, 1],
                            padding=padding
                            )
        conv = tf.layers.batch_normalization(conv)
        if activatiton:
            return tf.nn.relu(conv + biases, name='output_activated')
        else:
            return tf.identity(conv + biases, name='output_non_activated')
