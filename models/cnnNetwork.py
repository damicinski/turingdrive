import tensorflow as tf
import numpy as np
from models.components import netComp as nc

class CnnNetwork():
    def __init__(self, inputSize, outputSize):
        self._batch_size = 25
        self._learningRate = 1e-2
        self._inputSize = list(inputSize)
        self._outputSize = list(outputSize)
        self._PH_x = tf.placeholder(tf.float32, [None] + self._inputSize, name = 'x')
        self._PH_keepProbability = tf.placeholder(tf.float32, name = 'keepProbability')
        self._PH_y_ = tf.placeholder(tf.float32, [None] + self._outputSize, name = 'y_')

    def buildNetwork(self):

        layers = []
        reluLeakyness = 0.1

        # Convolution 1
        with tf.name_scope('Convolution_1') as scope:
            input_depth = self._PH_x.get_shape().as_list()[3]
            filterDepth = 8
            filter_shape = [3, 3, input_depth, filterDepth]
            stride = 1
            conv1 = nc.conv_layer(self._PH_x, filter_shape, stride, reluLeakyness, padding = "SAME")
            layers.append(conv1)

        #Convolution 2
        with tf.name_scope('Convolution_2') as scope:
            input_depth = layers[-1].get_shape().as_list()[3]
            filterDepth = 16
            stride = 1
            filter_shape = [3, 3, input_depth, filterDepth]
            conv2 = nc.conv_layer(inpt, filter_shape, stride, reluLeakyness, padding = "SAME")
            layers.append(conv2)

        # Pooling 1
        with tf.name_scope('Pooling_1') as scope:
            pool1 = nc.pooling_layer(layers[-1],2)
            layers.append(pool1)

        # Convolution 3
        with tf.name_scope('Convolution_3') as scope:
            input_depth = layers[-1].get_shape().as_list()[3]
            filterDepth = 32
            stride = 1
            filter_shape = [3, 3, input_depth, filterDepth]
            conv3 = nc.conv_layer(inpt, filter_shape, stride, reluLeakyness, padding = "SAME")
            layers.append(conv3)

        # Pooling 1
        with tf.name_scope('Pooling_2') as scope:
            pool2 = nc.pooling_layer(layers[-1],2)
            layers.append(pool2)

        #Reshape
        with tf.name_scope('Reshape') as scope:
            p2Size = np.prod(np.array(h_pool2.get_shape().as_list()[1:]))
            pool2_flat = tf.reshape(h_pool2, [-1, p2Size])
            layers.append(pool2_flat)

        # Dropout
        with tf.name_scope('Dropout1'):
            drop1 = tf.nn.dropout(h_pool2_flat,self._PH_keepProbability)
            layers.append(drop1)

        # Fully connected layer
        with tf.name_scope('Fully_connected_1') as scope:
            fc1 = fc_layer(layers[-1],1024, reluLeakyness, flat = False,linear = False)
            layers.append(fc1)

        # Fully connected layer
        with tf.name_scope('Fully_connected_2') as scope:
            fc2 = fc_layer(layers[-1],self._outputSize, reluLeakyness, flat = False, linear = True)
            layers.append(fc2)

        self._y = layers[-1]

        #Mean square error
        self._mse = tf.reduce_sum(tf.pow(self._PH_y_-self._y,2))/(2*self._batch_size)
        #AdamOptimizer minimizing mean square error
        self._train_step = tf.train.AdamOptimizer(self._learningRate).minimize(self._mse)
