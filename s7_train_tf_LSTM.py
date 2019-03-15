import pydevd
import tensorflow as tf
import numpy as np

import util
from s3_image_batches_generator import ImageBatchGenerator
from s4_pre_trained_vgg16 import vgg_16, load_npy_weights
# from s4_pre_trained_mobilenet_v1 import mobilenet_v1_base
from s6_tf_LSTM import tf_lstm

pydevd.settrace('192.168.0.167', port=18236, stdoutToServer=True, stderrToServer=True)

categories = util.CATEGORIES
lstm_layers = util.LSTM_LAYER
frame_num = util.FRAME_NUM
hidden_layer_nodes = util.HIDDEN_LAYER_NODES
base_lr = util.LEARNING_RATE
batch_size = util.BATCH_SIZE

lr_decay_steps = util.LR_DECAY_STEPS
lr_decay_rate = util.LR_DECAY_RATE
weight_decay = util.WEIGHT_DECAY
dropout_keep_prob = util.DROPOUT_KEEP_PROB

# Input data.
spatial_size = [240, 320]
down_sampling_factor = 2
# one_img_squeeze_length = train_batch.spatial_size[0] * train_batch.spatial_size[1] // (
#         train_batch.down_sampling_factor ** 2)
one_img_squeeze_length = 512


# checkpoint_path = 'vgg_16_2016_08_28/vgg_16.ckpt'
# checkpoint_path = 'mobilenet_v1_0.5_160/mobilenet_v1_0.5_160.ckpt'


def add_placeholders_ops():
    with tf.name_scope('placeholders'):
        inputs_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, spatial_size[0] // down_sampling_factor, spatial_size[1] // down_sampling_factor, 3],
            name='inputs_placeholder')

        labels_placeholder = tf.placeholder(tf.float32, shape=[None, len(categories)], name='labels_placeholder')

        return inputs_placeholder, labels_placeholder


def add_pre_trained_cnn_ops(inputs_placeholder):
    # net shape: [batch_size * frame_num , 4, 5, one_img_squeeze_length]
    net = vgg_16(inputs_placeholder, one_img_squeeze_length, scope='vgg_16')
    # net shape: [batch_size * frame_num , 2, 3, one_img_squeeze_length]
    # net, _ = mobilenet_v1_base(inputs_placeholder, depth_multiplier=0.5, scope='MobilenetV1')

    # net shape: [batch_size * frame_num , one_img_squeeze_length]
    net = tf.nn.max_pool(net, ksize=[1, 4, 5, 1], strides=[1, 4, 5, 1],
                         padding='VALID', name='max_pool_cnn_outputs')

    # net shape after reshape: [batch_size, frame_num, one_img_squeeze_length]
    net = tf.reshape(net, [-1, frame_num, one_img_squeeze_length], name='reshape_vgg16_outputs')
    # net shape after transpose: [frame_num, batch_size, one_img_squeeze_length]
    net = tf.transpose(net, [1, 0, 2], name='transpose_vgg16_outputs')

    return net


def add_lstm_ops(net):
    with tf.name_scope('LSTM'):
        outputs, state = tf_lstm(net, lstm_layers, hidden_layer_nodes, batch_size, dropout_keep_prob)
        return outputs


def add_training_ops(outputs, labels_placeholder, wd):
    with tf.name_scope('classifier'):
        # Classifier.
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([hidden_layer_nodes, len(categories)], 0.0, 1.0), name='w')
        b = tf.Variable(tf.zeros([len(categories)]), name='b')

        # The size of logits: [batch_size, len(categories)]
        logits = tf.nn.xw_plus_b(outputs[-1], w, b, name='logits')
        tf.summary.histogram('logits', logits)

        # LSTM final predictions, [batch_size, len(categories)].
        final_prediction = tf.nn.softmax(logits, name='lstm_final_prediction')

        if wd is not None:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for variable in variables:
                weight_decay_value = tf.multiply(tf.nn.l2_loss(variable), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay_value)

        cross_entropy_mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=labels_placeholder, logits=logits), name='cross_entropy_mean')
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
        tf.add_to_collection('losses', cross_entropy_mean)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

    # with tf.name_scope('loss_averages'):
    #     # Compute the moving average of all individual losses and the total loss.
    #     loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #     losses = tf.get_collection('losses')
    #     loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.name_scope('optimizer'):
        # Optimizer.
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(
            base_lr, global_step, lr_decay_steps, lr_decay_rate,
            staircase=True, name='learning_rate_exponential_decay')
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate, name='AdamOptimizer')
        # optimizer = tf.train.AdagradOptimizer(learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        gradients, v = zip(*optimizer.compute_gradients(total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25, name='clip_by_global_norm')
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step, name='apply_gradients')

    # # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # with tf.control_dependencies([optimizer, variables_averages_op]):
    #     train_op = tf.no_op(name='train')

    return learning_rate, total_loss, optimizer, final_prediction


def add_evaluation_step(final_prediction, labels_placeholder):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # the size of prediction: [batch_size]
            prediction = tf.argmax(final_prediction, 1, name='prediction')
            # the size of correct_prediction: [batch_size], elements type is bool.
            correct_prediction = tf.equal(
                prediction, tf.argmax(tf.cast(labels_placeholder, tf.int64), 1), name='correct_prediction')
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='evaluation_step')
            tf.summary.scalar('accuracy', evaluation_step)

    return evaluation_step


def main():
    # ImageBatchGenerator.
    train_batch = ImageBatchGenerator('./data/train/train_list.txt', frame_num, spatial_size, categories,
                                      down_sampling_factor=down_sampling_factor, shuffle=True)
    test_batch = ImageBatchGenerator('./data/test/test_list.txt', frame_num, spatial_size, categories,
                                     down_sampling_factor=down_sampling_factor, shuffle=False)

    # Placeholders.
    inputs_placeholder_op, labels_placeholder_op = add_placeholders_ops()

    # Restore pre-trained CNN model from checkpoint file.
    net = add_pre_trained_cnn_ops(inputs_placeholder_op)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    session = tf.Session()
    # saver = tf.train.Saver()
    # saver.restore(session, checkpoint_path)
    load_npy_weights(session)

    # Simple LSTM.
    outputs_op = add_lstm_ops(net)

    # Add training ops.
    learning_rate_op, total_loss_op, optimizer_op, final_prediction_op = add_training_ops(outputs_op,
                                                                                          labels_placeholder_op,
                                                                                          weight_decay)
    # Add evaluation ops.
    evaluation_step_op = add_evaluation_step(final_prediction_op, labels_placeholder_op)

    # merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()

    variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(variable)
    variable_name = [v.name for v in tf.trainable_variables()]
    print(variable_name)

    num_steps = 10001
    summary_frequency = 100

    with session as sess:
        # write summaries out to the summaries_path
        train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
        test_writer = tf.summary.FileWriter('./summary/test')

        tf.global_variables_initializer().run()
        print('Initialized')

        for step in range(num_steps):
            # train_batch_clips shape:
            # [batch_size, frame_num, spatial_size[0]//down_sampling_factor, spatial_size[1]//down_sampling_factor, 3]
            train_batch_clips, train_batch_labels = train_batch.next_batch(batch_size)
            train_batch_clips = np.reshape(train_batch_clips, [-1, spatial_size[0] // down_sampling_factor,
                                                               spatial_size[1] // down_sampling_factor, 3])

            feed_dict = dict()
            feed_dict[inputs_placeholder_op] = train_batch_clips
            feed_dict[labels_placeholder_op] = train_batch_labels

            _, l, accuracy, lr, train_summary = sess.run(
                [optimizer_op, total_loss_op, evaluation_step_op, learning_rate_op, merged],
                feed_dict=feed_dict)

            if step % 50 == 0:
                train_writer.add_summary(train_summary, step)
                print('step: %d,      loss=%f,      acc=%f%%,  lr=%f,' % (step, l, accuracy * 100, lr))

            if step % summary_frequency == 0:
                test_batch_clips, test_batch_labels = test_batch.next_batch(batch_size)
                test_batch_clips = np.reshape(test_batch_clips, [-1, spatial_size[0] // down_sampling_factor,
                                                                 spatial_size[1] // down_sampling_factor, 3])

                feed_dict[inputs_placeholder_op] = test_batch_clips
                feed_dict[labels_placeholder_op] = test_batch_labels

                test_l, test_accuracy, test_summary = sess.run([total_loss_op, evaluation_step_op, merged],
                                                               feed_dict=feed_dict)
                test_writer.add_summary(test_summary, step)
                print('          test_loss=%f, test_acc=%f%%' % (test_l, test_accuracy * 100))

        # # save checkpoints.
        # if not os.path.exists(MODEL_SAVE_PATH):
        #     os.makedirs(MODEL_SAVE_PATH)
        # saver.save(sess, MODEL_SAVE_PATH + "/" + MODEL_NAME, global_step=global_step)


if __name__ == '__main__':
    main()
