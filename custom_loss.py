
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def ranking_loss(margin):
    def operation(x, label=0):
        x = tf.math.add(margin, x)
        x = K.maximum(x, 0)
        y = tf.reduce_sum(x, axis=1)  # Sum of errors in a batch | Shape: [batch_size]
        z = tf.reduce_mean(y)  # Average of all the elements in the batch
        return z
    return operation


def similarity_loss(margin):
    def operation(x, label=0):
        x = tf.math.abs(x)
        x = tf.math.subtract(x, margin)
        x = K.maximum(x, 0)
        y = tf.reduce_sum(x, axis=1)  # Sum of errors in a batch | Shape: [batch_size]
        z = tf.reduce_mean(y)  # Average of all the elements in the batch
        return z
    return operation


@tf.autograph.experimental.do_not_convert
def custom_loss(margin, beta):
    @tf.autograph.experimental.do_not_convert
    def operation(output, label=0):
        ranking_output = output[0]
        similarity_output = output[1]
        r_loss = ranking_loss(margin)(ranking_output)
        s_loss = similarity_loss(margin)(similarity_output)
        # print("r_loss: {} | s_loss: {}".format(r_loss.numpy(), s_loss.numpy()))
        return (tf.math.multiply(beta, r_loss) +
                    tf.math.multiply((1-beta), s_loss))
    return operation

