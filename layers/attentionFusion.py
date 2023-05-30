import tensorflow as tf

class AttentionFusion(tf.keras.layers.Layer):
    def __init__(self, d_model, n_channels, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.d_model = d_model

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.wq = tf.keras.layers.Dense(self.n_channels * self.d_model)
        self.wk = tf.keras.layers.Dense(self.n_channels * self.d_model)
        self.wv = tf.keras.layers.Dense(self.n_channels * self.d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, 1, self.n_channels, self.d_model))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.expand_dims(x, axis=1)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaling = dk ** -0.5
        scaled_attention_logits = matmul_qk * scaling

        v = tf.reshape(v, (batch_size, self.n_channels, self.d_model))

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.reshape(attention_weights, (batch_size, -1, self.n_channels))
        scaled_attention = tf.reshape(tf.matmul(attention_weights, v),(batch_size, self.d_model))

        return self.layernorm(scaled_attention), attention_weights

