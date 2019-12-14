import tensorflow as tf


class Model:

    def __init__(self,
                 seq_len,
                 class_num,
                 dropout_ratio=0.5,
                 output_chord_emb=16,
                 rnn_unit=128,
                 dense_unit=32):

        self.seq_len = seq_len
        self.class_num = class_num
        self.dropout_ratio = dropout_ratio
        self.output_chord_emb = output_chord_emb
        self.rnn_unit = rnn_unit
        self.dense_unit = dense_unit

    def placeholders(self):
        with tf.name_scope('input'):
            input_pl = tf.placeholder(tf.float32, (None, self.seq_len), name="input")
            target_pl = tf.placeholder(tf.int32, (None), name="target")
        return input_pl, target_pl

    def loss(self, pred, labels):
        with tf.name_scope('loss'):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=pred)
            tf.summary.scalar('loss', loss)
        return loss

    def optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(loss)

    def infer(self, inputs, is_training):

        x = tf.keras.layers.Embedding(input_dim=self.class_num,
                                      output_dim=self.output_chord_emb,
                                      input_length=self.seq_len,
                                      name='chord_embedding',
                                      )(inputs)

        # x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_unit))(x)
        x = tf.keras.layers.GRU(self.rnn_unit)(x)
        x = tf.keras.layers.Dense(self.class_num, activation=None)(x)
        x = tf.identity(x, name="output")
        return x