
import csv
import cv2
import numpy as np
import os

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


DATASET = "/proj/xsjhdstaff4/akorra/datasets/videos"
optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()


def extract_frames(sequence_dirs, save_path, resize=None, num_processes=1):
    output_by_input = []
    for sequence_dir in sequence_dirs:
        for video in os.listdir(sequence_dir):
            output_dir = os.path.join(save_path, os.path.basename(sequence_dir),
                                      os.path.splitext(video)[0])
            video_path = os.path.join(sequence_dir, video)
            output_by_input.append((video_path, output_dir))
    func = partial(process, resize)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(func, output_by_input),
                            total=len(output_by_input)))


def process(resize, output_by_input):
    input_video_file, output_dir = output_by_input
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_file)
    length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = 0
    end_frame = length_video

    # Only useful for DoughRolling dataset
    frames_to_consider = {}
    with open('EPIC-Skills2018/dough_rolling_segments.csv', 'r') as file:
        frames_to_consider = {row[0]: (row[1], row[2])
                              for row in csv.reader(file)}
    vfile_basename = os.path.basename(input_video_file)
    for key, value in frames_to_consider.items():
        if key in vfile_basename:
            start_frame, end_frame = value
            break

    prev = None
    for frame_idx in range(end_frame):
        ret, rgb_frame = cap.read()
        if (frame_idx < start_frame):
            continue
        rgb_file = os.path.join(output_dir, "rgb_{:04d}.npy".format(frame_idx+1))
        flow_file = os.path.join(output_dir, "flow_{:04d}.npy".format(frame_idx+1))
        if resize:
            rgb_frame = cv2.resize(rgb_frame, (resize, resize))
        np.save(rgb_file, rgb_frame)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = optical_flow.calc(prev, gray, None)
            np.save(flow_file, flow)
        prev = gray
    cap.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', "--dataset-folder", default=DATASET)
    parser.add_argument('-s', '--sequences', required=True, nargs='+')
    parser.add_argument('-n', '--num_processes', default=5, type=int)
    parser.add_argument('-r', '--resize', default=256, type=int)
    parser.add_argument('-o', '--output-folder', default="frames")
    args = parser.parse_args()
    # sequences = ["DoughRolling", "ChopstickUsing", "HandDrawing",
    #              "SonicDrawing", "Knot_Tying", "Needle_Passing", "Suturing"]
    seq_dirs = [os.path.join(args.dataset_folder, seq) for seq in args.sequences]
    extract_frames(seq_dirs, args.output_folder, resize=args.resize,
                   num_processes=args.num_processes)

