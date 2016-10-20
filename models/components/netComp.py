import tensorflow as tf
import numpy as np

def weight_variable(shape, name = 'W'):
    initial = tf.truncated_normal(shape, stddev = 0.1, name = name)
    return tf.Variable(initial)

def bias_variable(shape, name = 'b'):
    initial = tf.constant(0.1, shape = shape, name = name)
    return tf.Variable(initial)

def pooling_layer(inputs,size):
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, size, size, 1], padding='SAME',name='pool')

def fc_layer(inputs,hiddens,leakyness = 0.1, flat = False,linear = False):
    input_shape = inputs.get_shape().as_list()
    if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_transposed = tf.transpose(inputs,(0,3,1,2))
        inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
    else:
        dim = input_shape[1]
        inputs_processed = inputs

    weight = weight_variable([dim,hiddens])
    biases = bias_variable([hiddens])

    if linear:
        return tf.add(tf.matmul(inputs_processed,weight),biases,name = 'fc')

    ip = tf.add(tf.matmul(inputs_processed,weight),biases)
    return tf.maximum(leakyness*ip,ip,name = 'leaky_relu')

def conv_layer(inpt, filter_shape, stride, leakyness = 0.1, padding = "SAME"):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding=padding)
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    return tf.maximum(leakyness*batch_norm,batch_norm,name='leaky_relu')


def residual_block(inpt, output_depth, down_sample, leakyness = 0.1, projection=False, padding="SAME"):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1, leakyness, padding)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1, leakyness, padding)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2, leakyness, padding)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    return conv2 + input_layer
