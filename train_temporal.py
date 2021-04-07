
import math
import numpy as np
import os
import tensorflow as tf

from argparse import ArgumentParser
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Concatenate, Input, Lambda, Subtract
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from time import time

from AlexNet import AlexNet
from custom_loss import custom_loss, ranking_loss, similarity_loss
from data_generator import get_temporal_dataset


NUM_SNIPPETS = 7
MARGIN = 1.0
BETA = 0.5
BATCH_SIZE = 128
MOMENTUM = 0.9  # gradient descent
TEMPORAL_LEARNING_RATE = 5e-3


def create_base_model(weights_path=None):
    model = AlexNet()
    if weights_path:
        model.load_weights(weights_path)
    # x = model.layers[-2].output
    # x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=model.input, outputs=model.output)


def create_model(num_snippets, num_input_channels, plot_model_as=None, margin=MARGIN):
    # inputs = []
    better_inputs = []
    worse_inputs = []
    ranking_outputs = []
    similarity_outputs = []
    base_model = create_base_model()
    for i in range(num_snippets):
        better_input_layer = Input(shape=(224, 224, num_input_channels),
                                   name='better_input_layer_{}'.format(i))
        worse_input_layer = Input(shape=(224, 224, num_input_channels),
                                  name='worse_input_layer_{}'.format(i))
        better_skill = base_model(better_input_layer)
        worse_skill = base_model(worse_input_layer)
        diff_ranking = Subtract(name='diff_ranking_{}'.format(i))([worse_skill, better_skill])
        diff_similarity = Subtract(name='diff_similarity_{}'.format(i))([better_skill, worse_skill])
        # inputs.extend([better_input_layer, worse_input_layer])
        better_inputs.append(better_input_layer)
        worse_inputs.append(worse_input_layer)
        ranking_outputs.append(diff_ranking)
        similarity_outputs.append(diff_similarity)
    inputs = better_inputs + worse_inputs
    r_outputs = Lambda(lambda x: x)(ranking_outputs.pop())
    s_outputs = Lambda(lambda x: x)(similarity_outputs.pop())
    while ranking_outputs:
        r_outputs = Concatenate()([r_outputs, ranking_outputs.pop()])
        s_outputs = Concatenate()([s_outputs, similarity_outputs.pop()])
    r_outputs = Lambda(lambda x: x, name="ranking_concat")(r_outputs)
    s_outputs = Lambda(lambda x: x, name="similarity_concat")(s_outputs)
    outputs = [r_outputs, s_outputs]
    model = Model(inputs=inputs, outputs=outputs)
    if plot_model_as:
        plot_model(model, to_file=plot_model_as, show_shapes=True,
                   show_layer_names=True)
    return model


@tf.function
def train_step(model, optimizer, margin, beta, batch):
    with tf.GradientTape() as tape:
        y = model(batch, training=True)
        r_loss = ranking_loss(margin)(y[0])
        s_loss = similarity_loss(margin)(y[1])
        losses = [r_loss, s_loss]
    model_gradient = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
    return model_gradient, losses


def validate_batch(model, margin, beta, batch):
    y = model(batch, training=False)
    val_loss = custom_loss(margin, beta)(y)
    return val_loss


def set_learning_rate(optimizer, iteration):
    if (iteration > 0) and (iteration % 10000 == 0 or iteration % 16000 == 0):
        optimizer.lr = optimizer.lr * 0.1


def train_temporal_model(num_snippets, num_iterations, learning_rate,
                        batch_size=128, momentum=0.9, margin=1.0, beta=0.5):
    model = create_model(num_snippets=num_snippets, num_input_channels=10)
    train_dataset = get_temporal_dataset("train.csv", batch_size, num_snippets)
    val_dataset = get_temporal_dataset("val.csv", 2, num_snippets)
    optimizer = Adam(lr=learning_rate)
    epochs = max(1, math.ceil(num_iterations/batch_size))
    iteration = 0
    for epoch in range(epochs):
        for train_batch in train_dataset:
            set_learning_rate(optimizer, iteration)
            if iteration == num_iterations:
                break
            start_time = time()
            model_gradient, losses = train_step(model, optimizer, margin, beta, train_batch)
            iteration += 1
            print("Train step: {}/{} | Loss: {:.3f}/{:.3f} | Time taken: {:.3f} s"
                  .format(iteration, num_iterations, losses[0], losses[1], time()-start_time))
        # for val_batch in val_dataset:
        #     val_loss = validate_batch(model, margin, beta, val_batch)
        #     print("Val loss: {:.3f} / {:.3f}".format(val_loss[0], val_loss[1]))
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--snippets", type=int, default=NUM_SNIPPETS)
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-i", "--iterations", type=int, default=18000)
    parser.add_argument("-lm", "--margin", type=float, default=MARGIN)
    parser.add_argument("-lb", "--beta", type=float, default=BETA)
    parser.add_argument("-lr", "--learning-rate", type=float, default=TEMPORAL_LEARNING_RATE)
    parser.add_argument("-m", "--momentum", type=float, default=MOMENTUM)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    train_temporal_model(args.snippets, args.iterations, args.learning_rate,
                         args.batch_size, args.momentum, args.margin, args.beta)

