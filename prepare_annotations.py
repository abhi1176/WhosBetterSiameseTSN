
import os
import pandas as pd

from argparse import ArgumentParser
from glob import glob


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a", "--annotations-dir", default="EPIC-Skills2018/annotations")
    parser.add_argument('-f', '--frames', default='frames')
    args = parser.parse_args()

    columns = ["Better", "Worse"]

    for split in range(1, 5):
        all_train_df = pd.DataFrame()
        all_val_df = pd.DataFrame()
        for sequence in os.listdir(args.annotations_dir):
            print("[INFO] Processing Split: {} of {}...".format(split, sequence))

            similar_pairs_csv = os.path.join(args.annotations_dir, sequence, "similar_pairs.csv")
            if not os.path.exists(similar_pairs_csv):
                print("[Skipping] {} does not exist..".format(similar_pairs_csv))
                continue

            sequence_dir = os.path.join(args.annotations_dir, sequence, "splits")
            train_csv = glob(os.path.join(sequence_dir, "*_train_{}.csv".format(split)))[0]
            val_csv = glob(os.path.join(sequence_dir, "*_val_{}.csv".format(split)))[0]

            train_df = pd.read_csv(train_csv, usecols=columns, index_col=None)
            train_df = train_df.applymap(lambda x: os.path.join(args.frames, sequence, x))
            train_df['label'] = pd.Series([1] * train_df.shape[0])
            # print(train_df.head())

            similar_df = pd.read_csv(similar_pairs_csv, index_col=None)
            similar_df = similar_df.applymap(lambda x: os.path.join(args.frames, sequence, x))
            similar_df['label'] = pd.Series([0] * similar_df.shape[0])

            val_df = pd.read_csv(val_csv, usecols=columns)
            val_df = val_df.applymap(lambda x: os.path.join(args.frames, sequence, x))
            val_df['label'] = pd.Series([1] * val_df.shape[0])

            all_train_df = all_train_df.append(train_df)
            all_train_df = all_train_df.append(similar_df)
            all_val_df = all_val_df.append(val_df)

        all_train_df = all_train_df.sample(frac=1)
        os.makedirs("split_{}".format(split), exist_ok=True)
        train_file = "split_{}/train.csv".format(split)
        val_file = "split_{}/val.csv".format(split)
        print("[INFO] Saving {} to {}".format(all_train_df.shape, train_file))
        print("[INFO] Saving {} to {}\n".format(all_val_df.shape, val_file))
        all_train_df.to_csv(train_file, index=False)
        all_val_df.to_csv(val_file, index=False)
