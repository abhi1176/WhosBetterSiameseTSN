
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from glob import glob
from skimage.transform import resize
from tensorflow.keras.models import load_model, Model


def get_snippets(seq_dir, num_snippets=7, rgb_pattern="rgb_*", flow_pattern="flow_*",
                 stack_depth=5, clip=None):
    rgb_files = sorted(list(glob(os.path.join(seq_dir, rgb_pattern))))
    flow_files = sorted(list(glob(os.path.join(seq_dir, flow_pattern))))

    num_files = len(flow_files)
    rgb_blob = []
    flow_blob = []
    boundaries = np.linspace(0, num_files//2, num_snippets+1, dtype=int)
    for first, second in zip(boundaries, boundaries[1:]):
        choice = np.random.randint(first, second-stack_depth)
        flow_stack = []
        for i in range(stack_depth):
            flow_data = np.load(flow_files[choice+i]).astype(np.float32)
            flow_data = resize(flow_data, (224, 224))
            if clip:
                flow_data = ((flow_data + clip)/(2*clip))*255.0
            flow_data = flow_data - np.mean(flow_data, axis=(0, 1))
            flow_stack.append(flow_data)
        rgb_data = np.load(rgb_files[choice+i+1]).astype(np.float32)
        rgb_data = resize(rgb_data, (224, 224))
        rgb_blob.append(rgb_data - np.mean(rgb_data, axis=(0, 1)))
        flow_blob.append(np.concatenate(np.asarray(flow_stack), axis=2))
    return rgb_blob, flow_blob


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-tm", "--temporal-model", required=True)
    parser.add_argument("-sm", "--spatial-model", required=True)
    parser.add_argument('-i', "--input-file", default='test.csv')
    parser.add_argument('-b', "--batch-size", default=64, type=int)
    parser.add_argument('-s', "--snippets", default=7, type=int)
    parser.add_argument('-a', "--alpha", default=0.5, type=float)
    args = parser.parse_args()
    temporal_model = load_model(args.temporal_model)
    spatial_model = load_model(args.spatial_model)
    df = pd.read_csv(args.input_file)
    df = df[df['label'] == 1]
    temporal_model = temporal_model.get_layer(name='model')
    temporal_model = Model(inputs=temporal_model.inputs, outputs=temporal_model.outputs)

    spatial_model = spatial_model.get_layer(name='model')
    spatial_model = Model(inputs=spatial_model.inputs, outputs=spatial_model.outputs)
    positive = negative = 0
    for index, row in df.iterrows():
        better_rgb_blob, better_flow_blob = get_snippets(row['Better'], num_snippets=args.snippets)
        worse_rgb_blob, worse_flow_blob = get_snippets(row['Worse'], num_snippets=args.snippets)

        for s_input, t_input in zip(better_rgb_blob, better_flow_blob):
            s_output = spatial_model(np.expand_dims(s_input, 0))
            t_output = temporal_model(np.expand_dims(t_input, 0))
            better_score = args.alpha * s_output + (1-args.alpha) * t_output

        for s_input, t_input in zip(worse_rgb_blob, worse_flow_blob):
            s_output = spatial_model(np.expand_dims(s_input, 0))
            t_output = temporal_model(np.expand_dims(t_input, 0))
            worse_score = args.alpha * s_output + (1-args.alpha) * t_output
        if better_score > worse_score:
            positive += 1
        else:
            negative += 1
        print("Better score: {} | Worse score: {} | verdict: {}".format(
              better_score, worse_score, better_score > worse_score))
    print("Accurarcy: {}".format(positive/(positive+negative)))
'''
python evaluate.py -sm spatial_models/spatial_model_iter_100 -tm temporal_models/temporal_model_iter_001
'''