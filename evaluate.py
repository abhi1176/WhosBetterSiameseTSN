
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from glob import glob
from skimage.transform import resize
from tensorflow.keras.models import load_model, Model

from data_generator import get_spatial_dataset
from model_utils import create_model



if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("-tm", "--temporal-model", required=True)
    parser.add_argument("-sm", "--spatial-model", required=True)
    parser.add_argument('-i', "--input-file", default='test.csv')
    parser.add_argument('-b', "--batch-size", default=64, type=int)
    parser.add_argument('-s', "--snippets", default=7, type=int)
    parser.add_argument('-a', "--alpha", default=0.5, type=float)
    args = parser.parse_args()

    print("[INFO] Preparing Spatial Model: {}".format(args.spatial_model))
    s_model = create_model(num_snippets=args.snippets, num_input_channels=3)
    s_model.load_weights(args.spatial_model)
    time_distributed = s_model.get_layer(name='time_distributed')
    o1 = time_distributed(s_model.inputs[0])
    o2 = time_distributed(s_model.inputs[1])
    spatial_model = Model(inputs=s_model.inputs, outputs=[o1, o2])
    spatial_model.summary()

    # print("[INFO] Preparing Temporal Model: {}".format(args.temporal_model))
    # t_model = create_model(num_snippets=args.snippets, num_input_channels=10)
    # t_model.load_weights(args.temporal_model)
    # time_distributed = t_model.get_layer(name='time_distributed')
    # o1 = time_distributed(t_model.inputs[0])
    # o2 = time_distributed(t_model.inputs[1])
    # temporal_model = Model(inputs=t_model.inputs, outputs=[o1, o2])
    # temporal_model.summary()

    print("[INFO] Preparing the dataset..")
    test_dataset = get_spatial_dataset(args.input_file, args.batch_size, args.snippets)

    positive = negative = 0
    test_iterator = iter(test_dataset)

    num_batches = 742//args.batch_size
    for i in range(num_batches):
        print("[INFO] {}/{}: Running with batch_size: {}"
              .format(i, num_batches, args.batch_size))    
        X, y = test_iterator.get_next()
        better_scores, worse_scores = spatial_model(X)
        for b_snippets_scores, w_snippets_scores in zip(better_scores, worse_scores):
            b_score = np.sum(b_snippets_scores)
            w_score = np.sum(w_snippets_scores)
            if b_score > w_score:
                positive += 1
            else:
                negative += 1
            print("Better score: {} | Worse score: {} | verdict: {}".format(
                  b_score, w_score, b_score > w_score))
        print("Accurarcy: {}".format(positive/(positive+negative)))

'''
python evaluate.py -b 742 -sm spatial_models_timedistributed/spatial_model_iter_190.h5 -tm temporal_models/temporal_model_iter_001
'''
