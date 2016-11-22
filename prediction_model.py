import tensorflow as tf

# import prediction_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x = tf.placeholder(tf.float32, shape=[1, 480, 640, 3])
# y_ = tf.placeholder(tf.float32, shape=[None, 1])
# x = read_data.train_image_batch
# y_ = read_data.train_angle_batch
keep_prob = tf.placeholder(tf.float32)

x_image = x
# x_image = tf.cast(x, tf.float32)

# Construct the first convolutional layer
W_conv1 = weight_variable([7, 7, 3, 24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 4) + b_conv1)

# Construct the second convolutional layer
W_conv2 = weight_variable([7, 7, 24, 36])
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 4) + b_conv2)

# Third convolutional layer
W_conv3 = weight_variable([7, 7, 36, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 4) + b_conv3)

# Fourth convolutional layer
W_conv4 = weight_variable([5, 5, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

# Fifth convolutional layer
W_conv5 = weight_variable([2, 2, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

# First fully connected layer
W_fc1 = weight_variable([1*4*64, 1164/5])
b_fc1 = bias_variable([1164/5])

h_conv5_flat = tf.reshape(h_conv5, [-1, 1*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Second fully connected layer
W_fc2 = weight_variable([1164/5, 100])
b_fc2 = bias_variable([100])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Third fully connected layer
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# Fourth fully connected layer
W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable([10])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

# Output control layer
W_fc5 = weight_variable([10, 1])
b_fc5 = bias_variable([1])

y_conv = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
