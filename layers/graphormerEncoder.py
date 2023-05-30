import  tensorflow as tf
from .multiHeadAttention import MultiHeadAttention

class GraphormerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, d_sp_enc=128, sp_enc_activation='relu', **kwargs):
        super(GraphormerBlock, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)


    def call(self, x, training, mask, min_distance_matrix):
        residual = x
        x_norm = self.layernorm1(x)
        attn_output, _, = self.mha(x_norm, min_distance_matrix, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = residual + attn_output

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = residual + ffn_output

        return out2, _


def point_wise_feed_forward_network(d_model, dff):

    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
