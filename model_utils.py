
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from alexnet_func_cpu import tsn_alexnet
# from alexnet_func_gpu import tsn_alexnet


MARGIN = 1
BETA = 0.5

@tf.autograph.experimental.do_not_convert
def get_custom_loss(outputs, y):
    y = tf.cast(y, tf.float32)  # y = [1 for distinguishable_pair, 0 if similar_pair]
    ranking_loss = K.mean(y * K.maximum(0.0, MARGIN - outputs[0] + outputs[1]))
    similarity_loss = K.mean((1.0-y) * K.maximum(0.0, K.abs(outputs[0] - outputs[1]) - MARGIN))
    return (tf.math.multiply(BETA, ranking_loss) +
                tf.math.multiply((1-BETA), similarity_loss))


def get_accuracy(scores):
    positive = 0
    for b_snippets_scores, w_snippets_scores in zip(scores[0], scores[1]):
        b_score = np.sum(b_snippets_scores)
        w_score = np.sum(w_snippets_scores)
        if b_score > w_score:
            positive += 1
    return positive/len(scores[0])


def create_model(num_snippets, num_input_channels, plot_model_as=None):
    base_model = tsn_alexnet(input_shape=(224, 224, num_input_channels))
    # base_model.summary()

    time_distributed = TimeDistributed(base_model)

    better_inputs = Input(shape=(num_snippets, 224, 224, num_input_channels))
    better_outputs = time_distributed(better_inputs)

    worse_inputs = Input(shape=(num_snippets, 224, 224, num_input_channels))
    worse_outputs = time_distributed(worse_inputs)

    model = Model(inputs=[better_inputs, worse_inputs], outputs=[better_outputs, worse_outputs])
    # model.summary()
    try:
        if plot_model_as:
            plot_model(model, to_file=plot_model_as, show_shapes=True,
                       show_layer_names=True)
    except Exception as e:
        print("[EXCEPTION] Unable to plot model..")
        print(e)
    return model


if __name__ == "__main__":
    spatial_model = create_model(num_snippets=7, num_input_channels=3,
                                 plot_model_as="spatial_model.png")
    temporal_model = create_model(num_snippets=7, num_input_channels=10,
                                 plot_model_as="temporal_model.png")
