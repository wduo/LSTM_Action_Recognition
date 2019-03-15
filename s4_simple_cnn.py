# import pydevd
import tensorflow as tf
from tensorflow.python.ops import variable_scope
import numpy as np

import util
from s3_image_batches_generator import ImageBatchGenerator


# pydevd.settrace('192.168.0.105', port=10098, stdoutToServer=True, stderrToServer=True)


def vgg_16(inputs, one_img_squeeze_length=4096, dropout_keep_prob=0.5, name='VGG_16'):
    with variable_scope.variable_scope(name, 'vgg_16', [inputs]):
        net = tf.layers.conv2d(inputs, filters=8, kernel_size=[3, 3],
                               padding='SAME', activation=tf.nn.relu, name='conv1_0')
        # net = tf.layers.conv2d(inputs, filters=64, kernel_size=[3, 3], padding='SAME', name='conv1_1')
        # net = tf.layers.conv2d(net, 64, [3, 3], padding='SAME', name='conv1_2')
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        net = tf.layers.conv2d(net, 8, [3, 3], padding='SAME', activation=tf.nn.relu, name='conv2_0')
        # net = tf.layers.conv2d(net, 128, [3, 3], padding='SAME', name='conv2_1')
        # net = tf.layers.conv2d(net, 128, [3, 3], padding='SAME', name='conv2_2')
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')

        net = tf.layers.conv2d(net, 16, [3, 3], padding='SAME', activation=tf.nn.relu, name='conv3_0')
        # net = tf.layers.conv2d(net, 256, [3, 3], padding='SAME', name='conv3_1')
        # net = tf.layers.conv2d(net, 256, [3, 3], padding='SAME', name='conv3_2')
        # net = tf.layers.conv2d(net, 256, [3, 3], padding='SAME', name='conv3_3')
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')

        net = tf.layers.conv2d(net, 16, [3, 3], padding='SAME', activation=tf.nn.relu, name='conv4_0')
        # net = tf.layers.conv2d(net, 512, [3, 3], padding='SAME', name='conv4_1')
        # net = tf.layers.conv2d(net, 512, [3, 3], padding='SAME', name='conv4_2')
        # net = tf.layers.conv2d(net, 512, [3, 3], padding='SAME', name='conv4_3')
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool4')

        # net = tf.layers.conv2d(net, 512, [3, 3], padding='SAME', name='conv5_1')
        # net = tf.layers.conv2d(net, 512, [3, 3], padding='SAME', name='conv5_2')
        # net = tf.layers.conv2d(net, 512, [3, 3], padding='SAME', name='conv5_3')
        # net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool5')

        # Use conv2d instead of fully_connected layers.
        net = tf.layers.conv2d(net, one_img_squeeze_length, [8, 10], activation=tf.nn.relu, name='fc6')
        net = tf.nn.dropout(net, dropout_keep_prob, name='dropout6')
        net = tf.layers.conv2d(net, one_img_squeeze_length, [1, 1], activation=tf.nn.relu, name='fc7')

    return net


def main():
    categories = util.CATEGORIES
    frame_num = util.FRAME_NUM
    batch_size = util.BATCH_SIZE

    spatial_size = [240, 320]
    # ImageBatchGenerator.
    train_batch = ImageBatchGenerator('./data/train/train_list.txt', frame_num, spatial_size, categories,
                                      down_sampling_factor=1, shuffle=True)

    input_placeholder = tf.placeholder(tf.float32, shape=[None, spatial_size[0], spatial_size[1], 3])

    net = vgg_16(input_placeholder)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        train_batch_clips, train_batch_labels = train_batch.next_batch(batch_size)
        train_batch_clips = np.reshape(train_batch_clips, [-1, spatial_size[0], spatial_size[1], 3])
        for i in range(10):
            net_value = sess.run([net], feed_dict={input_placeholder: train_batch_clips})
            # print(net_value)


if __name__ == '__main__':
    main()
