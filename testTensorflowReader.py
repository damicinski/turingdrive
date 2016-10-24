
# =============================================================================
# 					A quick test of how tfreader works in practice 
#
# Code from tensorflow tutorial using mnist and a single hidden layer fc 
# network is modified to support a tfrecords reader instead of the feed_dict 
# method. 
# =============================================================================

import tensorflow as tf
import numpy as np

# =============================================================================
# 					READ AND DECODE 
# =============================================================================
def read_and_decode_single_example(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([10], tf.int64),
            'image': tf.FixedLenFeature([784], tf.int64)
        })
    # now return the converted data
    label = features['label']
    image = features['image']

    label = tf.cast(label, tf.int32)
    # Set image range between -0.5 and 0.5 
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return label, image
# =============================================================================
#
# =============================================================================

# get single examples
label, image = read_and_decode_single_example("mnist_test.tfrecords")

# groups examples into batches randomly
# More info on how to se capacity and min_after_dequeue is necessary
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=500,
    capacity=2000,
    min_after_dequeue=1000)

# Implementing the regression

'''
Placeholders are used for our input data, and will later be supplanted with input data. The first argument is type, and the second one is the dimension of the data. In this case, None corresponds to variable number of rows, and we have 784 columns. 
'''
#x = tf.placeholder(tf.float32, [None, 784])
x = images_batch

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training 
'''
Placeholder for the correct answers, used to calculate the error between prediction and true value. 
'''
#y_ = tf.placeholder(tf.float32, [None, 10])
y_ = labels_batch

'''
Cross-entropy loss function.  reduce_sum calculates the sum along the dimension specified by reduction_indicies, in this case the second dimension of y. reduce_mean calculates the mean over all of the examples in the batch. 
However, this is not used when running since it is nymerically unstable
'''
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indicies=[1]))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

'''
Use Gradient Descent with learning-rate 0.5
'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
Define an operation to initialize all of the variables. They are not yet initialized however 
'''
init = tf.initialize_all_variables()

'''
Now the model is launced in a Session, which enables us to acctually initilazie the variables. 
'''

sess = tf.Session()
sess.run(init)

# Without these 2 lines, tensorflow will hang since it will be waiting 
# forever to receive data. 
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Let's train 
'''
Run the training step 1000 times. At each iteration of the loop, we get at "batch" of 100 random data points from our training set. We run train_step feeding in the batches data to replace our placeholders --> x & y_
'''

for i in range(1000):
#	batch_xs, batch_ys = mnist.train.next_batch(100)
	print i
#	sess.run(train_step, feed_dict = {x: images_batch, y_: labels_batch})
	sess.run(train_step)

# Evaluating our model 

'''
tf.argmax picks the index of max value along the specified dimension, which is this case for each column. Since values are probabilities, we get which number we predict, and compare that to the real value. 
'''

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Finally, print out the accuracy of the test data
'''

#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print(sess.run(accuracy))

coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()





