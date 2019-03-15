import tensorflow as tf
from tensorflow.python.ops import rnn_cell


def lstm_cell(hidden_layer_nodes, dropout_keep_prob):
    lstm_cell = rnn_cell.BasicLSTMCell(hidden_layer_nodes, state_is_tuple=True)
    lstm_cell = rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=dropout_keep_prob)
    return lstm_cell


def tf_lstm(inputs, layers, hidden_layer_nodes, batch_size, dropout_keep_prob):
    mlstm_cell = rnn_cell.MultiRNNCell([lstm_cell(hidden_layer_nodes, dropout_keep_prob) for _ in range(layers)],
                                       state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=inputs, initial_state=init_state, time_major=True)

    return outputs, state
