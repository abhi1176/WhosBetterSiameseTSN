
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from functools import partial
from glob import glob
from skimage.transform import resize


IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])


def get_snippets(seq_dir, num_snippets, pattern, stack_depth, clip=None):
    dirpath = os.path.join(seq_dir, pattern)
    files = sorted(list(glob(dirpath)))
    num_files = len(files)
    blob = []
    boundaries = np.linspace(0, num_files, num_snippets+1, dtype=int)
    for first, second in zip(boundaries, boundaries[1:]):
        choice = np.random.randint(first, second-stack_depth)
        stack = []
        for i in range(stack_depth):
            data = np.load(files[choice+i]).astype(np.float32)
            data = resize(data, (224, 224))
            if clip and pattern == "flow_*":
                data = ((data + clip)/(2*clip))*255.0
            else:
                data = data - IMAGENET_MEAN
            stack.append(data)
        stack = np.asarray(stack)
        stack = np.concatenate(stack, axis=2)
        blob.append(stack)
    return blob


def get_rgb_snippets(seq_dir, num_snippets, pattern="rgb_*", stack_depth=1):
    return get_snippets(seq_dir, num_snippets, pattern, stack_depth=stack_depth)


def get_flow_snippets(seq_dir, num_snippets, pattern="flow_*", stack_depth=5, clip=15):
    return get_snippets(seq_dir, num_snippets, pattern, stack_depth=stack_depth, clip=15)


def snippets_generator(csv_file, num_snippets, snippets_creator):
    def process():
        df = pd.read_csv(csv_file)
        df = df.sample(frac=1)  # Shuffle
        for index, row in df.iterrows():
            # print("[INFO] Generating the snippets for row: {} | Better: {} | Worse: {}"
            #       .format(index, row['Better'], row['Worse']))
            better_rgb_snippets = snippets_creator(row['Better'], num_snippets)
            worse_rgb_snippets = snippets_creator(row['Worse'], num_snippets)
            yield tuple(better_rgb_snippets + worse_rgb_snippets)
    return process


def dataset_generator(csv_file, batch_size, num_snippets, snippets_creator):
    gen = snippets_generator(csv_file, num_snippets, snippets_creator)
    dataset = tf.data.Dataset.from_generator(gen, output_types=tuple([tf.float32]*2*num_snippets))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset


def get_spatial_dataset(csv_file, batch_size, num_snippets):
    return dataset_generator(csv_file, batch_size, num_snippets, get_rgb_snippets)


def get_temporal_dataset(csv_file, batch_size, num_snippets):
    return dataset_generator(csv_file, batch_size, num_snippets, get_flow_snippets)
