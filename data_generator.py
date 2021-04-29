
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from functools import partial
from glob import glob
from skimage.transform import resize


IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])


def get_rgb_snippets(seq_dir, num_snippets):
    dirpath = os.path.join(seq_dir, "rgb_*")
    files = sorted(list(glob(dirpath)))
    num_files = len(files)
    blob = np.empty((num_snippets, 224, 224, 3))
    boundaries = np.linspace(0, num_files-1, num_snippets+1, dtype=int)
    for idx, (first, second) in enumerate(zip(boundaries, boundaries[1:])):
        choice = np.random.randint(first, second)
        data = np.load(files[choice]).astype(np.float32)
        data = resize(data, (224, 224))
        data = data - np.mean(data, axis=(0, 1))
        blob[idx, ...] = data
    return blob


def rgb_snippets_generator(csv_file, num_snippets):
    def process():
        df = pd.read_csv(csv_file)
        while True:
            sample_df = df.sample(1)
            row = sample_df.iloc[0, :]
            better_rgb_snippets = get_rgb_snippets(row['Better'], num_snippets)
            worse_rgb_snippets = get_rgb_snippets(row['Worse'], num_snippets)
            labels = row['label']
            yield (better_rgb_snippets, worse_rgb_snippets), labels
    return process


def get_spatial_dataset(csv_file, batch_size, num_snippets):
    gen = rgb_snippets_generator(csv_file, num_snippets)
    dataset = tf.data.Dataset.from_generator(
        gen, output_types=((tf.float32, tf.float32), tf.int32))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def get_flow_snippets(seq_dir, num_snippets, stack_depth=5, clip=15):
    dirpath = os.path.join(seq_dir, "flow_*")
    files = sorted(list(glob(dirpath)))
    num_files = len(files)
    blob = np.empty((num_snippets, 224, 224, 2*stack_depth))
    boundaries = np.linspace(0, num_files-1, num_snippets+1, dtype=int)
    for idx, (first, second) in enumerate(zip(boundaries, boundaries[1:])):
        choice = np.random.randint(first, second-stack_depth)
        for i in range(stack_depth):
            data = np.load(files[choice+i]).astype(np.float32)
            data = resize(data, (224, 224))
            if clip:
                data = ((data + clip)/(2*clip))*255.0
            data = data - np.mean(data, axis=(0, 1))
            blob[idx, ..., 2*i:2*(i+1)] = data
    return blob


def flow_snippets_generator(csv_file, num_snippets):
    def process():
        df = pd.read_csv(csv_file)
        while True:
            sample_df = df.sample(1)
            row = sample_df.iloc[0, :]
            better_flow_snippets = get_flow_snippets(row['Better'], num_snippets)
            worse_flow_snippets = get_flow_snippets(row['Worse'], num_snippets)
            labels = row['label']
            yield (better_flow_snippets, worse_flow_snippets), labels
    return process


def get_temporal_dataset(csv_file, batch_size, num_snippets):
    gen = flow_snippets_generator(csv_file, num_snippets)
    dataset = tf.data.Dataset.from_generator(
        gen, output_types=((tf.float32, tf.float32), tf.int32))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
