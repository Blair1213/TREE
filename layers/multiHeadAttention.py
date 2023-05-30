import tensorflow as tf
from .spatialEncoding import SpatialEncoding


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(self.d_sp_enc, self.sp_enc_activation)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, min_distance_matrix, mask):

        batch_size = tf.shape(q)[0]

        spatial_encoding_bias = self.spatial_encoding(min_distance_matrix)  ##[-1, seq_len_q, seq_len_k]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(spatial_encoding_bias)  # (batch_size, seq_len, d_model)
        v = self.wv(spatial_encoding_bias)  # (batch_size, seq_len, d_model)


        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.num_heads, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)


        return output, attention_weights

def scaled_dot_product_attention(q, k, v, num_heads, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaling = dk ** -0.5
    scaled_attention_logits = matmul_qk * scaling
    
    # add the mask to the scaled tensor. mask [bz, 1, 1, 5, 64]
    if mask is not None:
        scaled_attention_logits += (tf.repeat(mask, num_heads, axis=1) * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=3)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

