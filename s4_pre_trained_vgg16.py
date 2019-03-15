import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np


def _variable_on_cpu(name, shape):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, trainable=False)
    return var


def conv_op(inputs, kernal, strides, padding, scope):
    with tf.variable_scope(scope):
        w = _variable_on_cpu('weights', kernal)
        b = _variable_on_cpu('biases', kernal[-1])
        conv = tf.nn.conv2d(inputs, w, (1, strides[0], strides[1], 1), padding=padding, name=scope)
        z = tf.nn.bias_add(conv, b)
        activation = tf.nn.relu(z)
    return activation


def max_pool_op(inputs, strides, scope, kernal=(2, 2)):
    return tf.nn.max_pool(inputs, ksize=[1, kernal[0], kernal[1], 1],
                          strides=[1, strides[0], strides[1], 1],
                          padding='SAME', name=scope)


def vgg_16(inputs, one_img_squeeze_length, scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        with tf.variable_scope('conv1'):
            net = conv_op(inputs, kernal=[3, 3, 3, 64], strides=[1, 1], padding='SAME', scope='conv1_1')
            # net = conv_op(net, kernal=[3, 3, 64, 64], strides=[1, 1], padding='SAME', scope='conv1_2')
        net = max_pool_op(net, strides=[2, 2], scope='pool1')

        with tf.variable_scope('conv2'):
            net = conv_op(net, kernal=[3, 3, 64, 128], strides=[1, 1], padding='SAME', scope='conv2_1')
            # net = conv_op(net, kernal=[3, 3, 128, 128], strides=[1, 1], padding='SAME', scope='conv2_2')
        net = max_pool_op(net, strides=[2, 2], scope='pool2')

        with tf.variable_scope('conv3'):
            net = conv_op(net, kernal=[3, 3, 128, 256], strides=[1, 1], padding='SAME', scope='conv3_1')
            # net = conv_op(net, kernal=[3, 3, 256, 256], strides=[1, 1], padding='SAME', scope='conv3_2')
            # net = conv_op(net, kernal=[3, 3, 256, 256], strides=[1, 1], padding='SAME', scope='conv3_3')
        net = max_pool_op(net, strides=[2, 2], scope='pool3')

        with tf.variable_scope('conv4'):
            net = conv_op(net, kernal=[3, 3, 256, 512], strides=[1, 1], padding='SAME', scope='conv4_1')
            # net = conv_op(net, kernal=[3, 3, 512, 512], strides=[1, 1], padding='SAME', scope='conv4_2')
            # net = conv_op(net, kernal=[3, 3, 512, 512], strides=[1, 1], padding='SAME', scope='conv4_3')
        net = max_pool_op(net, strides=[2, 2], scope='pool4')

        with tf.variable_scope('conv5'):
            net = conv_op(net, kernal=[3, 3, 512, 512], strides=[1, 1], padding='SAME', scope='conv5_1')
        #     # net = conv_op(net, kernal=[3, 3, 512, 512], strides=[1, 1], padding='SAME', scope='conv5_2')
        #     # net = conv_op(net, kernal=[3, 3, 512, 512], strides=[1, 1], padding='SAME', scope='conv5_3')
        net = max_pool_op(net, strides=[2, 2], scope='pool5')

        # net = conv_op(net, kernal=[7, 7, 512, one_img_squeeze_length], strides=[1, 1], padding='VALID', scope='fc6')
        # net = conv_op(net, kernal=[1, 1, one_img_squeeze_length, one_img_squeeze_length],
        #               strides=[1, 1], padding='VALID', scope='fc7')

        return net


def load_npy_weights(session):
    weights_dict = np.load('vgg16.npy', encoding='latin1').item()
    load_weights = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    for op_name in load_weights:
        if op_name[0:4] == 'conv':
            prefix = op_name.split('_')[0] + '/'
        else:
            prefix = ''
        with tf.variable_scope('vgg_16/' + prefix + op_name, reuse=True):
            for data in weights_dict[op_name]:
                # Biases
                if len(data.shape) == 1:
                    var = tf.get_variable('biases')
                    session.run(var.assign(data))
                # Weights
                else:
                    var = tf.get_variable('weights')
                    session.run(var.assign(data))


def main_1():
    inputs = tf.ones([2, 120, 160, 3])

    net = vgg_16(inputs, 4096)

    session = tf.Session()
    load_npy_weights(session)

    with session as sess:
        tf.global_variables_initializer().run()

        net_value = sess.run([net])
        print(net_value)
        print(net_value[0].shape)


def main():
    checkpoint_path = 'vgg_16_2016_08_28/vgg_16.ckpt'

    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))  # Remove this if you want to print only variable names

    inputs = tf.ones([2, 120, 160, 3])

    net = vgg_16(inputs, 4096)
    net = tf.nn.max_pool(net, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1],
                         padding='SAME', name='max_pool_cnn_outputs')

    variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if variable:
        print(variable)
    else:
        print("NO TRAINABLE_VARIABLES.")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        tf.global_variables_initializer().run()

        net_value = sess.run([net])
        print(net_value)
        print(net_value[0].shape)


if __name__ == '__main__':
    main_1()
