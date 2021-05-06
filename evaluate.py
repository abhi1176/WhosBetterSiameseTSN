
import math
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
    parser.add_argument('-i', "--input-csv", required=True)
    parser.add_argument("-tm", "--temporal-model", required=True)
    parser.add_argument("-sm", "--spatial-model", required=True)
    parser.add_argument('-b', "--batch-size", default=32, type=int)
    parser.add_argument('-s', "--snippets", default=25, type=int)
    parser.add_argument('-a', "--alpha", default=0.4, type=float)
    args = parser.parse_args()

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

    df = pd.read_csv(args.input_csv)
    sequences = np.unique(df['Better'].apply(lambda x: os.path.dirname(x)).values)

    for sequence in sequences:
        seq_df = df[df['Better'].str.contains(sequence)]
        num_records = seq_df.shape[0]
        num_batches = math.ceil(num_records/args.batch_size)
        input_csv = os.path.join(os.path.dirname(args.input_csv),
                                 os.path.basename(sequence)) + ".csv"
        seq_df.to_csv(input_csv, index=False)

        print("[INFO] Preparing the dataset for {} from {}".format(sequence, input_csv))
        spatial_dataset = get_spatial_dataset(input_csv, args.batch_size, args.snippets, validation=True)
        temporal_dataset = get_temporal_dataset(input_csv, args.batch_size, args.snippets, validation=True)

        positive = negative = 0
        positive_spatial = 0
        negative_spatial = 0
        positive_temporal = 0
        negative_temporal = 0

        spatial_iterator = iter(spatial_dataset)
        temporal_iterator = iter(temporal_dataset)
        for i in range(num_batches):
            print("[INFO] {}/{}: Running with batch_size: {}"
                  .format(i+1, num_batches, args.batch_size))
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
        print("Input file: {}".format(input_csv))
        print("alpha: {}".format(args.alpha))
        print("Snippets: {}".format(args.snippets))
        print("Accurarcy: {:.3f}".format(positive/(positive+negative)))
        print("Spatial Accuracy: {:.3f}".format(positive_spatial/(positive_spatial+negative_spatial)))
        print("Temporal Accuracy: {:.3f}".format(positive_temporal/(positive_temporal+negative_temporal)))

'''
python evaluate.py \
    -sm models/split_1/spatial_model.h5 \
    -tm models/split_1/temporal_model.h5 \
    -i split_1/val.csv

python evaluate.py \
    -sm models/split_2/spatial_model.h5 \
    -tm models/split_2/temporal_model.h5 \
    -i split_2/val.csv
'''