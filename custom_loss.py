
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


@tf.autograph.experimental.do_not_convert
def get_custom_loss(margin, beta):
    def operation(outputs, y):
        ranking_loss = 0
        similarity_loss = 0
        y = tf.cast(y, tf.float32)  # y = [1 for distinguishable_pair, 0 if similar_pair]
        for i in range(0, len(outputs)//2):
            better_skill = outputs[2*i]
            worse_skill = outputs[2*i+1]
            # sig1 = tf.math.sigmoid(better_skill - worse_skill)
            # sig2 = tf.math.sigmoid(K.abs(better_skill - worse_skill))
            # ranking_loss += K.mean(y*K.maximum(0.0, margin - sig1))
            # similarity_loss += K.mean((1.0-y) * K.maximum(0.0, sig2 - margin))
            ranking_loss += y * K.maximum(0.0, margin - better_skill + worse_skill)
            similarity_loss += (1.0-y) * K.maximum(0.0, K.abs(better_skill - worse_skill) - margin)
        return (tf.math.multiply(beta, K.sum(ranking_loss)) +
                    tf.math.multiply((1-beta), K.sum(similarity_loss)))
    return operation
