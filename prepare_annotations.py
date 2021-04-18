
import csv
import os
import pandas as pd

from argparse import ArgumentParser
from glob import glob


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a", "--annotations-dir", default="EPIC-Skills2018/annotations",
        help="Directory containing sequence/splits/<train/val>.csv")
    parser.add_argument('-d', '--dataset', default='/proj/sdxapps/users/akorra/frames')
    args = parser.parse_args()

    columns = ["Better", "Worse"]
    dirname_by_seq = {"Chopstick_Using": "ChopstickUsing",
                      "Dough_Rolling": "DoughRolling",
                      "Hand_Drawing": "HandDrawing"}

    all_train_df = pd.DataFrame()
    all_val_df = pd.DataFrame()
    for sequence in os.listdir(args.annotations_dir):
        print("[INFO] Processing {}...".format(sequence))
        sequence_dir = os.path.join(args.annotations_dir, sequence, "splits")
        train_csvs = glob(os.path.join(sequence_dir, "*_train_*"))
        val_csvs = glob(os.path.join(sequence_dir, "*_val_*"))

        train_df = pd.DataFrame()
        val_df = pd.DataFrame()

        for train_csv in train_csvs:
            df = pd.read_csv(train_csv, usecols=columns)
            train_df = train_df.append(df)
        train_df = train_df.applymap(lambda x: os.path.join(
            args.dataset, dirname_by_seq.get(sequence, sequence), x))
        train_df['label'] = pd.Series([1] * train_df.shape[0])
        all_train_df = all_train_df.append(train_df)

        similar_df = pd.read_csv(
            os.path.join(args.annotations_dir, sequence, "similar_pairs.csv"))
        similar_df = similar_df.applymap(lambda x: os.path.join(
            args.dataset, dirname_by_seq.get(sequence, sequence), x))
        similar_df['label'] = pd.Series([0] * similar_df.shape[0])
        all_train_df = all_train_df.append(similar_df)

        for val_csv in val_csvs:
            df = pd.read_csv(val_csv, usecols=columns)
            val_df = val_df.append(df)
        val_df = val_df.applymap(lambda x: os.path.join(
            args.dataset, dirname_by_seq.get(sequence, sequence), x))
        all_val_df = all_val_df.append(val_df)
        all_val_df['label'] = pd.Series([1] * all_val_df.shape[0])

    all_train_df.to_csv("train.csv", index=False)
    all_val_df.to_csv("val.csv", index=False)
