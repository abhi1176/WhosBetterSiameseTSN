
import os
import numpy as np
import tensorflow as tf

from functools import partial
from glob import glob

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Concatenate, Subtract
from tensorflow.keras.utils import plot_model

from AlexNet import AlexNet


num_seq = 7
margin = tf.constant(1, dtype=tf.float32)
beta = 0.5
batch_size = 128
momentum = 0.9  # gradient descent
spatial_lr = 1e-3
temporal_lr = 5e-3


class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_train_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(batch, lr)
        # Set the value back to the optimizer before this batch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nbatch %05d: Learning rate is %6.4f." % (batch, scheduled_lr))


def spatial_lr_scheduler(batch, lr):
    if batch % 1500 == 0:
        lr = lr * 0.1
    return lr


def temporal_lr_scheduler(batch, lr):
    if (batch % 10000 == 0) or (batch % 16000 == 0):
        lr = lr * 0.1
    return lr


def get_snippets(seq_dir, num_seq, pattern, stack_depth):
    files = sorted(list(glob(os.path.join(seq_dir, pattern))))
    num_files = len(files)
    blob = []
    boundaries = np.linspace(0, num_files, num_seq+1, dtype=int)
    for first, second in zip(boundaries, boundaries[1:]):
        choice = np.random.randint(first, second-stack_depth)
        stack = []
        for i in range(stack_depth):
            stack.append(np.load(files[choice+i]))
        stack = np.asarray(stack)
        stack = np.concatenate(stack, axis=2)
        blob.append(stack)
    return blob


def get_rgb_snippets(seq_dir, num_seq, pattern="rgb_*", stack_depth=1):
    # num_seq x [h x w x 3]
    get_snippets(seq_dir, num_seq, pattern, stack_depth=stack_depth)


def get_flow_snippets(seq_dir, num_seq, pattern="flow_*", stack_depth=5):
    # num_seq x [h x w x 10]
    get_snippets(seq_dir, num_seq, pattern, stack_depth=stack_depth)


def process(num_seq, row):
    better_video_dir, worse_video_dir = row

    better_rgb_snippets = get_rgb_snippets(better_video_dir, num_seq)
    worse_rgb_snippets = get_rgb_snippets(worse_video_dir, num_seq)

    better_flow_snippets = get_flow_snippets(better_video_dir, num_seq)
    worse_flow_snippets = get_flow_snippets(worse_video_dir, num_seq)
    output = []
    output.extend(better_rgb_snippets)
    output.extend(worse_rgb_snippets)
    output.extend(better_flow_snippets)
    output.extend(worse_flow_snippets)
    return output


def data_generator(csv_files, batch_size, num_seq):
    dataset = tf.data.experimental.CsvDataset(
        csv_files, [tf.string, tf.string], header=True,
        select_cols=["Better", "Worse"])
    process_ = partial(process, num_seq)
    dataset = dataset.map(process_)
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_base_model():
    model = AlexNet()
    # model.load_weights("imagenet.h5")
    # x = model.layers[-2].output
    # x = Dense(1, activation='sigmoid')
    # return Model(inputs=model.input, outputs=model.output)
    return model



def create_spatial_model(num_seq):
    inputs = []
    ranking_outputs = []
    similarity_outputs = []
    base_model = create_base_model()
    for i in range(num_seq):
        better_input_layer = Input(shape=(224,224,3), name='better_input_layer{}'.format(i))
        worse_input_layer = Input(shape=(224,224,3), name='worse_input_layer{}'.format(i))

        better_skill = base_model(better_input_layer)
        worse_skill = base_model(worse_input_layer)

        diff_ranking = Subtract(name='diff_ranking_{}'.format(i))([worse_skill, better_skill])
        diff_similarity = Subtract(name='diff_similarity_{}'.format(i))([better_skill, worse_skill])

        inputs.extend([better_input_layer, worse_input_layer])
        ranking_outputs.append(diff_ranking)
        similarity_outputs.append(diff_similarity)

    ranking_concat = Concatenate(name="ranking_concat")(ranking_outputs)
    similarity_concat = Concatenate(name="similarity_concat")(similarity_outputs)
    model = Model(inputs=inputs, outputs=[ranking_concat, similarity_concat])
    plot_model(model, to_file='spatial_model.png', show_shapes=True, show_layer_names=True)
    return model


def run_spatial_model():
    model = create_spatial_model(num_seq)
    model.compile(loss=loss(margin, beta), optimizer=SGD(lr=spatial_lr, momentum=momentum))
    model.fit(spatial_train_gen, epochs=1, batch_size=batch_size,
              callbacks=[CustomLearningRateScheduler(spatial_lr_scheduler)])


def create_temporal_model(num_seq):
    inputs = []
    ranking_outputs = []
    similarity_outputs = []
    base_model = create_base_model()
    for i in range(num_seq):
        better_input_layer = Input(shape=(224,224,10), name='better_input_layer{}'.format(i))
        worse_input_layer = Input(shape=(224,224,10), name='worse_input_layer{}'.format(i))

        better_skill = base_model(better_input_layer)
        worse_skill = base_model(worse_input_layer)

        diff_ranking = Subtract(name='diff_ranking_{}'.format(i))([worse_skill, better_skill])
        diff_similarity = Subtract(name='diff_similarity_{}'.format(i))([better_skill, worse_skill])

        inputs.extend([better_input_layer, worse_input_layer])
        ranking_outputs.append(diff_ranking)
        similarity_outputs.append(diff_similarity)

    ranking_concat = Concatenate(name="ranking_concat")(ranking_outputs)
    similarity_concat = Concatenate(name="similarity_concat")(similarity_outputs)
    model = Model(inputs=inputs, outputs=[ranking_concat, similarity_concat])
    plot_model(model, to_file='temporal_model.png', show_shapes=True, show_layer_names=True)
    return model


def run_temporal_model():
    model = create_temporal_model(num_seq)
    model.compile(loss=loss(margin, beta), optimizer=SGD(lr=temporal_lr, momentum=momentum))
    model.fit(temporal_train_gen, epochs=1, batch_size=batch_size,
              callbacks=[CustomLearningRateScheduler(temporal_lr_scheduler)])


def ranking_loss(margin):
    def operation(x, label=0):
        x = tf.math.add(margin, x)
        x = tf.keras.backend.maximum(x, 0)
        return tf.reduce_sum(x)
    return operation


def similarity_loss(margin):
    def operation(x, label=0):
        x = tf.math.abs(x)
        x = tf.math.subtract(x, margin)
        x = tf.keras.backend.maximum(x, 0)
        return tf.reduce_sum(x)
    return operation


def loss(margin, beta):
    def operation(output, label=0):
        ranking_output = output[..., 0]
        similarity_output = output[..., 1]
        return (tf.math.multiply(beta, ranking_loss(margin)(ranking_output)) +
                    tf.math.multiply((1-beta), similarity_loss(margin)(similarity_output)))
    return operation

