
import logging
import math
import os
import tensorflow as tf

from argparse import ArgumentParser
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from time import time

from data_generator import get_temporal_dataset
from model_utils import create_model, get_custom_loss


logging.basicConfig(filename='log_train_temporal.log', level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_SNIPPETS = 21
NUM_ITERATIONS = 18000
BATCH_SIZE = 32
MOMENTUM = 0.9  # SGD
LEARNING_RATE = 5e-3
VALIDATION_BATCH_SIZE = 1

optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)


@tf.function
def train_step(model, batch):
    with tf.GradientTape() as tape:
        X, y = batch
        outputs = model(X, training=True)
        loss = get_custom_loss(outputs, y)
    model_gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
    return loss


def validate_batch(model, val_iterator):
    val_batch = val_iterator.get_next()
    X, y = val_batch
    outputs = model(X, training=False)
    for b_skill, w_skill in zip(outputs[0][0], outputs[1][0]):
        logger.info("{} x {} => {}".format(b_skill.numpy(), w_skill.numpy(), y[0]))
    val_loss = get_custom_loss(outputs, y)
    logger.info("Val loss: {:.3f}".format(val_loss))
    return val_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--snippets", type=int, default=NUM_SNIPPETS)
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-vb", "--validation-batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    parser.add_argument("-i", "--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--split", type=int, required=True)
    args = parser.parse_args()

    model = create_model(num_snippets=args.snippets, num_input_channels=10)
    train_file = os.path.join("split_{}/train.csv".format(args.split))
    val_file = os.path.join("split_{}/val.csv".format(args.split))
    train_dataset = get_temporal_dataset(train_file, args.batch_size, args.snippets)
    val_dataset = get_temporal_dataset(val_file, args.validation_batch_size, args.snippets)

    train_iterator = iter(train_dataset)
    val_iterator = iter(val_dataset)

    models_dir = "models/split_{}/temporal_models_{}".format(args.split, args.snippets)
    os.makedirs(models_dir, exist_ok=True)

    start_time = time()
    for iteration in range(1, args.iterations+1):
        if (iteration % 10000 == 0) or (iteration % 16000 == 0):
            K.set_value(optimizer.learning_rate, optimizer.learning_rate*0.1)
        train_batch = train_iterator.get_next()
        train_start = time()
        loss = train_step(model, train_batch)
        logger.info("Train step: {}/{} | loss: {:.3f} | train_step: {:.3f} s | loop: {:.3f} s"
              .format(iteration, args.iterations, loss, time()-train_start, time()-start_time))
        print("Train step: {}/{} | loss: {:.3f} | train_step: {:.3f} s | loop: {:.3f} s"
              .format(iteration, args.iterations, loss, time()-train_start, time()-start_time))
        if iteration % 10 == 0:
            val_loss = validate_batch(model, val_iterator)
            save_path = os.path.join(models_dir, "temporal_model_iter_{:04d}.h5".format(iteration))
            model.save_weights(save_path)
        start_time = time()

    val_loss = validate_batch(model, val_iterator)
    save_path = os.path.join(models_dir, "temporal_model_iter_last")
    model.save(save_path)
