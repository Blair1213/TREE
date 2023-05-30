import tensorflow as tf
from keras.regularizers import l2

class CentralityEncoding(tf.keras.layers.Layer):
    def __init__(self, max_degree, d_model, **kwargs):
        super(CentralityEncoding, self).__init__(**kwargs)
        self.centr_embedding = tf.keras.layers.Embedding(input_dim = max_degree, output_dim = d_model, name='centr_embedding')

    def centrality(self, distances):
        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), tf.float32)
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, tf.float32)

    def call(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.centr_embedding(centrality) ##[batch_size, n_graphs, n_nodes, 64]

        return tf.cast(centrality_encoding, tf.float32) ##[batch_size, n_graphs, n_nodes, 64]
