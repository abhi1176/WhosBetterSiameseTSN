
import math
import os

from argparse import ArgumentParser
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from time import time

from data_generator import get_spatial_dataset
from model_utils import create_model, train_step, validate_batch


NUM_SNIPPETS = 7
NUM_ITERATIONS = 3500
MARGIN = 1.0
BETA = 0.5
BATCH_SIZE = 128
MOMENTUM = 0.9  # SGD
LEARNING_RATE = 1e-3


def set_learning_rate(optimizer, iteration):
    if (iteration > 0) and (iteration % 1500 == 0):
        optimizer.lr = optimizer.lr * 0.1


def train_spatial_model(num_snippets=NUM_SNIPPETS, num_iterations=NUM_ITERATIONS,
                        learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                        momentum=MOMENTUM, margin=MARGIN, beta=BETA):
    model = create_model(num_snippets=num_snippets, num_input_channels=3,
                         plot_model_as='spatial_model.png')
    train_dataset = get_spatial_dataset("train.csv", batch_size, num_snippets)
    val_dataset = get_spatial_dataset("val.csv", 2, num_snippets)
    optimizer = SGD(lr=learning_rate)
    num_records = 5836
    num_iterations_per_epoch = math.ceil(num_records/batch_size)
    epochs = max(1, math.ceil(num_iterations/num_iterations_per_epoch))
    # epochs = max(1, math.ceil(num_iterations/batch_size))
    iteration = 0
    for epoch in range(epochs):
        print(" ====== Epoch: {}/{}".format(epoch+1, epochs))
        for train_batch in train_dataset:
            set_learning_rate(optimizer, iteration)
            if iteration == num_iterations:
                break
            start_time = time()
            model_gradient, losses = train_step(model, optimizer, margin, beta, train_batch)
            iteration += 1
            print("Train step: {}/{} | Loss: {:.3f}/{:.3f} | Time taken: {:.3f} s"
                  .format(iteration, num_iterations_per_epoch,
                          losses[0], losses[1], time()-start_time))
        # for val_batch in val_dataset:
        #     val_loss = validate_batch(model, margin, beta, val_batch)
        #     print("Val loss: {:.3f} / {:.3f}".format(val_loss[0], val_loss[1]))
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--snippets", type=int, default=NUM_SNIPPETS)
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-i", "--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("-lm", "--margin", type=float, default=MARGIN)
    parser.add_argument("-lb", "--beta", type=float, default=BETA)
    parser.add_argument("-lr", "--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("-m", "--momentum", type=float, default=MOMENTUM)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    train_spatial_model(args.snippets, args.iterations, args.learning_rate,
                        args.batch_size, args.momentum, args.margin, args.beta)
