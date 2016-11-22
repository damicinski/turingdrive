import csv
import os

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
NUM_CHANNELS = 3
BATCH_SIZE = 50

dataset_path = '/home/deepjedi/dat/udacity/challenge2/challenge2/train/21'
train_angle_file = os.path.join(dataset_path, 'center.csv')

with open(train_angle_file, 'rb') as csv_file:
    reader = csv.reader(csv_file)
    header = reader.next()
    train_images = []
    train_angles = []
    for line in reader:
        # print line
        train_images.append(line[0])
        train_angles.append(float(line[1]))

train_filepaths = [os.path.join(dataset_path, 'center', image + ".png") for image in train_images]

all_filepaths = train_filepaths
all_angles = train_angles

# train_images =
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_angles = ops.convert_to_tensor(all_angles, dtype=dtypes.float32)

train_images = all_images
train_angles = all_angles

# Create input queques
train_input_queque = tf.train.slice_input_producer([train_images, train_angles], shuffle=False)
# test_input_queque = tf.train.slice_input_producer([test_images, test_angles], shuffle=False)

# Process path and string tensors into image and label
file_content = tf.read_file(train_input_queque[0])
train_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
train_angle = train_input_queque[1]

# file_content = tf.read_file(test_input_queque[0])
# test_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
# test_angle = test_input_queque[1]

# Define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
# test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

# Collect batches of images before processing
train_image_batch, train_angle_batch = tf.train.batch([train_image, train_angle], batch_size=BATCH_SIZE)
# test_images_batch, test_angle_batch = tf.train.batch([test_image, test_angle], batch_size=BATCH_SIZE)
