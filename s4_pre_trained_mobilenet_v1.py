from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pydevd
from collections import namedtuple
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

slim = tf.contrib.slim

# pydevd.settrace('192.168.0.167', port=18236, stdoutToServer=True, stderrToServer=True)

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),  # This line was modified by wangduo.
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),  # This line was modified by wangduo.
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]


def _fixed_padding(inputs, kernel_size, rate=1):
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                    [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def mobilenet_v1_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      use_explicit_padding=False,
                      scope=None):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'
    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1

            net = inputs
            for i, conv_def in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i

                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel)
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params={'trainable': False},  # Add by wangduo.
                                      trainable=False,  # Add by wangduo.
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'

                    # By passing filters=None
                    # separable_conv2d produces only a depthwise convolution layer
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel, layer_rate)
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=layer_stride,
                                                rate=layer_rate,
                                                normalizer_fn=slim.batch_norm,
                                                normalizer_params={'trainable': False},  # Add by wangduo.
                                                trainable=False,  # Add by wangduo.
                                                scope=end_point)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                    end_point = end_point_base + '_pointwise'

                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params={'trainable': False},  # Add by wangduo.
                                      trainable=False,  # Add by wangduo.
                                      scope=end_point)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def main():
    # meta_path = 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt.meta'
    # checkpoint_dir_path = 'mobilenet_v1_1.0_224/'
    # checkpoint_path = 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
    # checkpoint_path = 'mobilenet_v1_0.5_160/mobilenet_v1_0.5_160.ckpt'
    checkpoint_path = 'mobilenet_v1_0.25_128/mobilenet_v1_0.25_128.ckpt'

    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))  # Remove this if you want to print only variable names

    inputs = tf.ones([2, 240, 320, 3])

    net, end_points = mobilenet_v1_base(inputs, depth_multiplier=0.25)

    # variable_name = [v.name for v in tf.trainable_variables()]
    # print(variable_name)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        tf.global_variables_initializer().run()

        net_value, _ = sess.run([net, end_points])
        print(net_value)
        print(net_value.shape)

    # sess = tf.Session()
    # saver = tf.train.import_meta_graph(meta_path)
    # saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_path))


if __name__ == '__main__':
    main()
