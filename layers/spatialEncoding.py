import tensorflow as tf

class SpatialEncoding(tf.keras.layers.Layer):
    def __init__(self, d_sp_enc=16, activation='relu'):
        super(SpatialEncoding, self).__init__()
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(d_sp_enc, activation=activation)
        self.dense2 = tf.keras.layers.Dense(64, activation=activation)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, distances):
        
        outputs = self.dense1(distances) ##[batch_size, n_nodes, d_sp_enc]
        outputs = self.dense2(outputs) ##[batch_size, n_nodes, 64]

        return outputs

