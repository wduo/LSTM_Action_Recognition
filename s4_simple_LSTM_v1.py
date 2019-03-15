import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops

import util


class SimpleLSTM:
    def __init__(self, num_unrollings, hidden_nodes, one_img_squeeze_length, n_classes, lstm_layers=1):
        """
        Simple LSTM Model.
        """
        self.num_unrollings = num_unrollings
        self.hidden_nodes = hidden_nodes
        self.one_img_squeeze_length = one_img_squeeze_length
        self.n_classes = n_classes
        self.lstm_layers = lstm_layers

        # Classifier weights and biases.
        self.w = tf.Variable(tf.truncated_normal([self.hidden_nodes, self.n_classes], 0.0, 1.0), name='w')
        self.b = tf.Variable(tf.zeros([self.n_classes]), name='b')

    def lstm_layer(self, inputs, layer_name, input_len, saved_state, saved_output, wd):
        """
        Single layer of LSTM.

        :return:
        """
        with variable_scope.variable_scope(layer_name):
            def variable_with_weight_decay(shape, name):
                with tf.device('/cpu:0'):
                    var = tf.Variable(tf.truncated_normal(shape), name=name)
                    # tf.summary.histogram(name, var)
                if wd is not None:
                    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
                    tf.add_to_collection('losses', weight_decay)
                return var

            # Input gate: input, previous output, and bias.
            ix = variable_with_weight_decay([input_len, self.hidden_nodes], name=layer_name + '_ix')
            im = variable_with_weight_decay([self.hidden_nodes, self.hidden_nodes], name=layer_name + '_im')
            ib = tf.Variable(tf.zeros([1, self.hidden_nodes]), name=layer_name + '_ib')

            # Forget gate: input, previous output, and bias.
            fx = variable_with_weight_decay([input_len, self.hidden_nodes], name=layer_name + '_fx')
            fm = variable_with_weight_decay([self.hidden_nodes, self.hidden_nodes], name=layer_name + '_fm')
            fb = tf.Variable(tf.zeros([1, self.hidden_nodes]), name=layer_name + '_fb')

            # Memory cell: input, state and bias.
            cx = variable_with_weight_decay([input_len, self.hidden_nodes], name=layer_name + '_cx')
            cm = variable_with_weight_decay([self.hidden_nodes, self.hidden_nodes], name=layer_name + '_cm')
            cb = tf.Variable(tf.zeros([1, self.hidden_nodes]), name=layer_name + '_cb')

            # Output gate: input, previous output, and bias.
            ox = variable_with_weight_decay([input_len, self.hidden_nodes], name=layer_name + '_ox')
            om = variable_with_weight_decay([self.hidden_nodes, self.hidden_nodes], name=layer_name + '_om')
            ob = tf.Variable(tf.zeros([1, self.hidden_nodes]), name=layer_name + '_ob')

            # # Variables saving state across unrollings.
            # saved_state = tf.Variable(tf.zeros([batch_size, self.hidden_nodes]), trainable=False,
            #                           name=layer_name + '_state')
            #
            # # Variables saving output across unrollings.
            # saved_output = tf.Variable(tf.zeros([batch_size, self.hidden_nodes]), trainable=False,
            #                            name=layer_name + '_o')

            def lstm_cell(i, o, state):
                input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib, name='input_gate')
                forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb, name='forget_gate')
                update = tf.add_n([tf.matmul(i, cx), tf.matmul(o, cm)], name='update') + cb
                # state = forget_gate * state + input_gate * nn_ops.relu(tf.tanh(update))
                state = tf.add(forget_gate * state, input_gate * tf.tanh(update), name='state')
                output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob, name='output_gate')
                # return output_gate * nn_ops.relu(tf.tanh(state)), state
                return output_gate * tf.tanh(state), state

            # def lstm_cell(i, o, state):
            #     input_gate = tf.sigmoid(tf.matmul(tf.concat([i, o], 1), ix) + ib)
            #     forget_gate = tf.sigmoid(tf.matmul(tf.concat([i, o], 1), fx) + fb)
            #     update = tf.matmul(tf.concat([i, o], 1), cx) + cb
            #     # state = forget_gate * state + input_gate * nn_ops.relu(tf.tanh(update))
            #     state = forget_gate * state + input_gate * tf.tanh(update)
            #     output_gate = tf.sigmoid(tf.matmul(tf.concat([i, o], 1), ox) + ob)
            #     # return output_gate * nn_ops.relu(tf.tanh(state)), state
            #     return output_gate * tf.tanh(state), state

            outputs_layer = []
            state = saved_state
            output = saved_output
            for ii in range(self.num_unrollings):
                output, state = lstm_cell(inputs[ii], output, state)
                outputs_layer.append(output)
            tf.summary.histogram('outputs_layer', outputs_layer)

        return outputs_layer, state

    def multi_layer_lstm(self, inputs, input_len, saved_state, saved_output, wd):
        """
        Multi layer LSTM.
        :param inputs:
        :param input_len:
        :param saved_state:
        :param saved_output:
        :param wd:
        :return:
        """
        # layer 1.
        final_outputs, final_state = self.lstm_layer(inputs=inputs, layer_name='layer_1', input_len=input_len,
                                                     saved_state=saved_state, saved_output=saved_output, wd=wd)

        # layer 2-n.
        if self.lstm_layers > 1:
            for layer in range(self.lstm_layers - 1):
                final_outputs, final_state = self.lstm_layer(inputs=final_outputs, layer_name='layer_' + str(layer + 2),
                                                             input_len=self.hidden_nodes,
                                                             saved_state=saved_state, saved_output=saved_output, wd=wd)

        return final_outputs, final_state


def main():
    hidden_layer_nodes = util.HIDDEN_LAYER_NODES
    frame_num = util.FRAME_NUM

    simple_lstm = SimpleLSTM(num_unrollings=frame_num, hidden_nodes=hidden_layer_nodes,
                             one_img_squeeze_length=240 * 320 // (8 ** 2), n_classes=6, lstm_layers=10)
    print(simple_lstm)


if __name__ == '__main__':
    main()
