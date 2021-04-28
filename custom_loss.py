
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def get_custom_loss(margin, beta):
    @tf.autograph.experimental.do_not_convert
    def operation(outputs, y):
        ranking_loss = 0
        similarity_loss = 0
        y = tf.cast(y, tf.float32)  # y = [1 for distinguishable_pair, 0 if similar_pair]
        for i in range(0, len(outputs)//2):
            better_skill = outputs[2*i]
            worse_skill = outputs[2*i+1]
            ranking_loss += K.mean(y * K.maximum(0.0, margin - better_skill + worse_skill))
            similarity_loss += K.mean((1.0-y) * K.maximum(0.0, K.abs(better_skill - worse_skill) - margin))
        return (tf.math.multiply(beta, ranking_loss) +
                    tf.math.multiply((1-beta), similarity_loss))
    return operation
