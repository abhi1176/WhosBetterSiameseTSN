
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from glob import glob
from skimage.transform import resize
from tensorflow.keras.models import load_model, Model

from data_generator import get_spatial_dataset, get_temporal_dataset
from model_utils import create_model



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-tm", "--temporal-model", required=True)
    parser.add_argument("-sm", "--spatial-model", required=True)
    parser.add_argument('-b', "--batch-size", default=64, type=int)
    parser.add_argument('-s', "--snippets", default=7, type=int)
    parser.add_argument('-a', "--alpha", default=0.4, type=float)
    parser.add_argument("--split", default=1, type=int)
    args = parser.parse_args()

    input_file = os.path.join("split_{}".format(args.split), "val.csv")
    print("[INFO] Preparing Spatial Model: {}".format(args.spatial_model))
    s_model = create_model(num_snippets=args.snippets, num_input_channels=3)
    s_model.load_weights(args.spatial_model)
    time_distributed = s_model.get_layer(name='time_distributed')
    o1 = time_distributed(s_model.inputs[0])
    o2 = time_distributed(s_model.inputs[1])
    spatial_model = Model(inputs=s_model.inputs, outputs=[o1, o2])
    # spatial_model.summary()

    print("[INFO] Preparing Temporal Model: {}".format(args.temporal_model))
    t_model = create_model(num_snippets=args.snippets, num_input_channels=10)
    t_model.load_weights(args.temporal_model)
    time_distributed = t_model.get_layer(name='time_distributed_1')
    o1 = time_distributed(t_model.inputs[0])
    o2 = time_distributed(t_model.inputs[1])
    temporal_model = Model(inputs=t_model.inputs, outputs=[o1, o2])
    # temporal_model.summary()

    print("[INFO] Preparing the dataset..")
    spatial_dataset = get_spatial_dataset(input_file, args.batch_size, args.snippets, shuffle=False)
    temporal_dataset = get_temporal_dataset(input_file, args.batch_size, args.snippets, shuffle=False)

    positive = negative = 0
    positive_spatial = 0
    negative_spatial = 0
    positive_temporal = 0
    negative_temporal = 0

    spatial_iterator = iter(spatial_dataset)
    temporal_iterator = iter(temporal_dataset)
    df = pd.read_csv(args.input_file)
    num_records = df.shape[0]
    num_batches = num_records//args.batch_size
    for i in range(num_batches):
        print("[INFO] {}/{}: Running with batch_size: {}"
              .format(i, num_batches, args.batch_size))
        spatial_X, y = spatial_iterator.get_next()
        temporal_X, y = temporal_iterator.get_next()
        better_scores, worse_scores = spatial_model(spatial_X)
        better_t_scores, worse_t_scores = temporal_model(temporal_X)
        for b_s_snippets_scores, w_s_snippets_scores, b_t_snippets_scores, w_t_snippets_scores in \
                zip(better_scores, worse_scores, better_t_scores, worse_t_scores):
            b_s_score = np.sum(b_s_snippets_scores)
            b_t_score = np.sum(b_t_snippets_scores)
            b_score = args.alpha*b_s_score + (1-args.alpha)*b_t_score

            w_s_score = np.sum(w_s_snippets_scores)
            w_t_score = np.sum(w_t_snippets_scores)
            w_score = args.alpha*w_s_score + (1-args.alpha)*w_t_score

            if b_score > w_score:
                positive += 1
            else:
                negative += 1

            if b_s_score > w_s_score:
                positive_spatial += 1
            else:
                negative_spatial += 1

            if b_t_score > w_t_score:
                positive_temporal += 1
            else:
                negative_temporal += 1
            # print("Better score: {} | Worse score: {} | verdict: {}".format(
            #       b_score, w_score, b_score > w_score))
    print("alpha: {}".format(args.alpha))
    print("Snippets: {}".format(args.snippets))
    print("Accurarcy: {:.3f}".format(positive/(positive+negative)))
    print("Spatial Accuracy: {:.3f}".format(positive_spatial/(positive_spatial+negative_spatial)))
    print("Temporal Accuracy: {:.3f}".format(positive_temporal/(positive_temporal+negative_temporal)))

