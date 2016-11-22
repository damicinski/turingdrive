import os

import tensorflow as tf

import read_data
import build_model

CKPT_DIR = './checkpoints'

NUM_EPOCH = 5

loss = tf.reduce_mean(tf.square(tf.sub(build_model.y_, build_model.y_conv)))
rmse = tf.sqrt(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver(max_to_keep=None)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

# Train over the dataset about 30 times
for i in range(len(read_data.train_filepaths)*NUM_EPOCH):
  xs, ys = read_data.train_image_batch, read_data.train_angle_batch
  train_step.run(feed_dict={build_model.keep_prob: 0.8})
  # sess.run(train_step)
  if i % 10 == 0:
    # xs, ys = read_data.LoadValBatch(100)
    print('Step: %d, RMSE: %g' % (i, rmse.eval(feed_dict={build_model.keep_prob: 1.0})))
  if i % 100 == 0:
    if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)
    ckpt_path = os.path.join(CKPT_DIR, "model%d.ckpt"%i)
    save_path = saver.save(sess, ckpt_path)
    print("Model saved in file: %s" % save_path)

coord.request_stop()
coord.join(threads)
sess.close()
