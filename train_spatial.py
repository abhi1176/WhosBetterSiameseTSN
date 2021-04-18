
import logging
import math
import os

from argparse import ArgumentParser
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from time import time
from tensorflow.keras import backend as K

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
    # better_skills, worse_skills = outputs
    # for b_skill, w_skill in zip(better_skills, worse_skills):
    #     print(b_skill, w_skill)
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

    num_records = 6157
    models_dir = "spatial_models"
    os.makedirs(models_dir, exist_ok=True)

    iters_per_epoch = math.ceil(num_records/args.batch_size)
    epochs = max(1, math.ceil(args.iterations/iters_per_epoch))

    iteration = 0
    for epoch in range(epochs):
        logger.info("====== Epoch: {}/{}".format(epoch+1, epochs))
        print("====== Epoch: {}/{}".format(epoch+1, epochs))
        start_time = time()
        for idx, train_batch in enumerate(train_dataset):
            if (iteration > 0) and (iteration % 1500 == 0):
                K.set_value(optimizer.learning_rate, optimizer.learning_rate*0.1)
            if iteration == args.iterations:
                break
            loss = train_step(model, train_batch)
            iteration += 1
            logger.info("Train step: {}/{} | Loss: {:.3f} | Time taken: {:.3f} s"
                  .format(idx, iters_per_epoch, loss, time()-start_time))
            print("Train step: {}/{} | Loss: {:.3f} | Time taken: {:.3f} s"
                  .format(idx, iters_per_epoch, loss, time()-start_time))
            start_time = time()
            for val_batch in val_dataset:
                val_loss = validate_batch(model, val_batch)
                logger.info("Val loss: {:.3f}".format(val_loss))
                print("Val loss: {:.3f}\n".format(val_loss))
                break
        save_path = os.path.join(models_dir, "spatial_model_epoch_{:03d}".format(epoch))
        model.save(save_path)
