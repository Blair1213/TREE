import tensorflow as tf
from keras import backend as K

class WeightedBinaryCrossEntropy(object):
    def __init__(self, pos_ratio):
        self.pos = pos_ratio
        self.neg = 1.0 - pos_ratio
        self.sample_weights = tf.cast(self.neg / self.pos, tf.float32)
        #self.__name__ = "weighted_binary_crossentropy({0})".format(self.sample_weights)

    def get_loss(self, y_true, y_pred):
        # Transform to logits
        y = tf.cast(y_true, tf.float32)

        epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.math.log(y_pred / (1 - y_pred))  ##做解码

        cost = tf.nn.weighted_cross_entropy_with_logits(y, y_pred, self.sample_weights)

        return K.mean(cost * self.pos, axis=-1)