
import logging
import math
import os
import tensorflow as tf

from argparse import ArgumentParser
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from time import time

from data_generator import get_spatial_dataset
from model_utils import create_model
from custom_loss import get_custom_loss


logging.basicConfig(filename='log_train_spatial.log', level=logging.INFO)
logger = logging.getLogger(__name__)


NUM_SNIPPETS = 7
NUM_ITERATIONS = 3500
MARGIN = 1.0
BETA = 0.5
BATCH_SIZE = 128
MOMENTUM = 0.9  # SGD
LEARNING_RATE = 1e-3
VALIDATION_BATCH_SIZE = 32

optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)
loss_fn = get_custom_loss(MARGIN, BETA)


@tf.function
def train_step(model, batch):
    with tf.GradientTape() as tape:
        X, y = batch
        outputs = model(X, training=True)
        loss = loss_fn(outputs, y)
    model_gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
    return loss


def validate_batch(model, batch):
    X, y = batch
    outputs = model(X, training=False)
    for i in range(len(outputs)//2):
        print(outputs[2*i][0].numpy(), "x", outputs[2*i+1][0].numpy())
    val_loss = loss_fn(outputs, y)
    return val_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--snippets", type=int, default=NUM_SNIPPETS)
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-vb", "--validation-batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    parser.add_argument("-i", "--iterations", type=int, default=NUM_ITERATIONS)
    args = parser.parse_args()

    model = create_model(num_snippets=args.snippets, num_input_channels=3,
                         plot_model_as='spatial_model.png')
    train_dataset = get_spatial_dataset("train.csv", args.batch_size, args.snippets)
    val_dataset = get_spatial_dataset("val.csv", args.validation_batch_size, args.snippets)

    train_iterator = iter(train_dataset)
    val_iterator = iter(val_dataset)

    models_dir = "spatial_models"
    os.makedirs(models_dir, exist_ok=True)

    start_time = time()
    for iteration in range(1, args.iterations+1):
        if iteration % 1500 == 0:
            K.set_value(optimizer.learning_rate, optimizer.learning_rate*0.1)
        train_batch = train_iterator.get_next()
        train_start = time()
        loss = train_step(model, train_batch)
        logger.info("Train step: {}/{} | loss: {:.3f} | train_step: {:.3f} s | loop: {:.3f} s"
              .format(iteration, args.iterations, loss, time()-train_start, time()-start_time))
        print("Train step: {}/{} | loss: {:.3f} | train_step: {:.3f} s | loop: {:.3f} s"
              .format(iteration, args.iterations, loss, time()-train_start, time()-start_time))
        if (iteration % 100 == 0) or iteration == args.iterations:
            val_batch = val_iterator.get_next()
            val_loss = validate_batch(model, val_batch)
            logger.info("Val loss: {:.3f}".format(val_loss))
            print("Val loss: {:.3f}\n".format(val_loss))

            save_path = os.path.join(models_dir, "spatial_model_iter_{:03d}".format(iteration))
            model.save(save_path)
        start_time = time()
