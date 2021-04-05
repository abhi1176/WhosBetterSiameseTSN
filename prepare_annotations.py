
import csv
import os
import pandas as pd

from argparse import ArgumentParser
from glob import glob


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a", "--annotations-dir",
        help="Directory containing sequence/splits/<train/val>.csv",
        default="EPIC-Skills2018/annotations")
    parser.add_argument('-d', '--dataset', default='frames')
    args = parser.parse_args()

    columns = ["Better", "Worse"]
    dirname_by_seq = {"Chopstick_Using": "ChopstickUsing",
                      "Dough_Rolling": "DoughRolling",
                      "Hand_Drawing": "HandDrawing"}

    all_train_df = pd.DataFrame(columns=columns)
    all_val_df = pd.DataFrame(columns=columns)
    for sequence in os.listdir(args.annotations_dir):
        print("[INFO] Processing {}...".format(sequence))
        sequence_dir = os.path.join(args.annotations_dir, sequence, "splits")
        train_csvs = glob(os.path.join(sequence_dir, "*_train_*"))
        val_csvs = glob(os.path.join(sequence_dir, "*_val_*"))

        train_df = pd.DataFrame(columns=columns)
        val_df = pd.DataFrame(columns=columns)

        for train_csv in train_csvs:
            df = pd.read_csv(train_csv, usecols=columns)
            train_df = train_df.append(df)
        train_df = train_df.applymap(lambda x: os.path.join(
            args.dataset, dirname_by_seq.get(sequence, sequence), x))
        all_train_df = all_train_df.append(train_df)

        for val_csv in val_csvs:
            df = pd.read_csv(val_csv, usecols=columns)
            val_df = val_df.append(df)
        val_df = val_df.applymap(lambda x: os.path.join(
            args.dataset, dirname_by_seq.get(sequence, sequence), x))
        all_val_df = all_val_df.append(val_df)

    all_train_df.to_csv("train.csv", index=False)
    all_val_df.to_csv("val.csv", index=False)

