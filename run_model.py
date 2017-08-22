import csv
import os

import cv2
import numpy as np
import scipy.misc
import tensorflow as tf

# import read_data
import prediction_model

from os import listdir
from os.path import splitext

CKPT_DIR = './checkpoints'
TEST_DIR = '/home/deepjedi/dat/udacity/challenge2/challenge2/test/center'

test_files = [f for f in listdir(TEST_DIR) if f.endswith('.png')]
test_files.sort()
# print 'huh'
# print listdir(TEST_DIR)
# print 'what'
# print test_files.sort()
# NUM_EPOCH = 100000

# loss = tf.reduce_mean(tf.square(tf.sub(build_model.y_, build_model.y_conv)))
# rmse = tf.sqrt(loss)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
# sess.run(init)

saver = tf.train.Saver()
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)
saver.restore(sess, 'checkpoints/model1000.ckpt')

dict_prediction = dict()

csv_file_path = '/home/deepjedi/dat/udacity/challenge2/challenge2/test/steering_angle_prediction.csv'

with open(csv_file_path, 'wb') as csv_file:
    field_names = ['frame_id', 'steering_angle']
    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    writer.writeheader()

    for file in test_files:
        # image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
        # print f
        img_name = os.path.splitext(file)[0]
        # print img_name
        img_path = os.path.join(TEST_DIR, file)
        # img = cv2.imread(f, flags=cv2.IMREAD_COLOR)
        img = cv2.imread(img_path)
        # cv2.imshow("steering wheel", img)
        # print 'Type: ', type(img)
        # print 'Shape: ', img.shape
        # img = scipy.misc.imresize(img, [480, 640, 3])
        # print img.shape

        angle_prediction = prediction_model.y_conv.eval(feed_dict={prediction_model.x: [img], prediction_model.keep_prob: 1.0})
        # print prediction_model.h_conv5_flat.eval(feed_dict={prediction_model.x: [img], prediction_model.keep_prob: 1.0})

        # rand_img = np.random.rand(480, 640, 3)
        # angle_prediction = prediction_model.y_conv.eval(feed_dict={prediction_model.x: [rand_img], prediction_model.keep_prob: 1.0})

        # zero_img = np.zeros((480, 640, 3))
        # angle_prediction = prediction_model.y_conv.eval(feed_dict={prediction_model.x: [zero_img], prediction_model.keep_prob: 1.0})
        # print prediction_model.h_fc4_drop.eval()
        dict_prediction = {'frame_id':img_name, 'steering_angle':float(angle_prediction)}
        print dict_prediction
        writer.writerow(dict_prediction)
        # angle_predictions[img_name] = angle_prediction
        # angle_predictions.append(angle_prediction)

# print angle_predictions
# csv_file = open('prediction.csv', 'wb')
# field_names = ['frame_id', 'steering_angle']
# writer = csv.DictWriter(csv_file, fieldnames=field_names)
# writer.writeheader()
# writer.writerows(angle_predictions)
# f.close()

# # Train over the dataset about 30 times
# for i in range(NUM_EPOCH):
#   xs, ys = read_data.train_image_batch, read_data.train_angle_batch
#   train_step.run(feed_dict={build_model.keep_prob: 0.8})
#   # sess.run(train_step)
#   if i % 10 == 0:
#     # xs, ys = read_data.LoadValBatch(100)
#     print("Step %d, Val Loss %g"%(i, rmse.eval(feed_dict={build_model.keep_prob: 1.0})))
#   if i % 100 == 0:
#     if not os.path.exists(CKPT_DIR):
#             os.makedirs(CKPT_DIR)
#     checkpoint_path = os.path.join(CKPT_DIR, "weights%d.ckpt"%i)
#     filename = saver.save(sess, checkpoint_path)
#     print("Weights saved in file: %s" % filename)

# coord.request_stop()
# coord.join(threads)
sess.close()
