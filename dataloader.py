
import os
import numpy as np
import tensorflow as tf

from glob import glob


def get_snippets(seq_dir, num_snippets, pattern, stack_depth):
	files = sorted(list(glob(os.path.join(seq_dir, pattern))))
	num_files = len(files)
	blob = []
	boundaries = np.linspace(0, num_files, num_snippets+1, dtype=int)
	for first, second in zip(boundaries, boundaries[1:]):
		choice = np.random.randint(first, second-stack_depth)
		stack = []
		for i in range(stack_depth):
			stack.append(np.load(files[choice+i]))
		stack = np.asarray(stack)
		stack = np.concatenate(stack, axis=2)
		blob.append(stack)
	return blob


def get_rgb_snippets(seq_dir, num_snippets, pattern="rgb_*", stack_depth=1):
	get_snippets(seq_dir, num_snippets, pattern, stack_depth=stack_depth)


def get_flow_snippets(seq_dir, num_snippets, pattern="flow_*", stack_depth=5):
	get_snippets(seq_dir, num_snippets, pattern, stack_depth=stack_depth)

def data_generator(csv_files):
	dataset = tf.data.experimental.CsvDataset(
		csv_files, [tf.string, tf.string], header=True,
		select_cols=["Better", "Worse"])
	
		

def create_base_model():
	model = AlexNet()
	model.load_weights("imagenet.h5")
	x = model.layers[-2].output
	x = Dense(1, activation='sigmoid')
	return Model(inputs=model.input, outputs=x)


def create_spatial_model(num_seq_per_video):
	inputs = []
	base_model = create_base_model()



# v1_s1: [batch_size, 1]

v1_concat = tf.keras.layers.Concatenate()([v1_s1, v1_s2, v1_s3, v1_s4, v1_s5, v1_s6, v1_s7])
v2_concat = tf.keras.layers.Concatenate()([v2_s1, v2_s2, v2_s3, v2_s4, v2_s5, v2_s6, v2_s7])
# v1_concat: [batch_size, 7]

difference_1 = tf.keras.layers.Subtract(axis=1)([v2_concat, v1_concat])
difference_2 = tf.keras.layers.Subtract(axis=1)([v1_concat, v2_concat])
# difference: [batch_size, 7]


margin = 1
def ranking_loss(margin):
	def operation(x, label=0):
		x = tf.math.add(margin, x)  # [batch_size, 7]
		x = tf.keras.backend.maximum(x, 0)  # [batch_size, 7]
		return tf.reduce_sum(x)


def similarity_loss(margin):
	def operation(x, label=0):
		x = tf.math.abs(x)  # [batch_size, 7]
		x = tf.math.subtract(x, m)  # [batch_size, 7]
		x = tf.keras.backend.maximum(x, 0)  # [batch_size, 7]
		return tf.reduce_sum(x)


