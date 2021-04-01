
import cv2
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()


def extract_frames(sequence_dirs, save_path, resize=None):
    output_by_input = []
    for sequence_dir in sequence_dirs:
        for video in os.listdir(sequence_dir):
            output_dir = os.path.join(save_path, os.path.basename(sequence_dir),
                                      os.path.splitext(video)[0])
            video_path = os.path.join(sequence_dir, video)
            output_by_input.append((video_path, output_dir))
    func = partial(process, resize)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, output_by_input),
                            total=len(output_by_input)))


def process(resize, output_by_input):
    input_video_file, output_dir = output_by_input
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video_file)
    length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 1
    prev = None
    while True:
        ret, rgb_frame = cap.read()
        if not ret:
            break
        rgb_file = os.path.join(output_dir, "rgb_{:04d}.npy".format(frame_idx))
        flow_file = os.path.join(output_dir, "flow_{:04d}.npy".format(frame_idx))
        if resize:
            rgb_frame = cv2.resize(rgb_frame, resize)
        np.save(rgb_file, rgb_frame)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        if frame_idx > 1:
            flow = optical_flow.calc(prev, gray, None)
            np.save(flow_file, flow)
        prev = gray
        frame_idx += 1


if __name__ == "__main__":
    dataset = "dataset"
    sequences = ['ChopstickUsing', "DoughRolling", "HandDrawing",
                 "Knot_Tying", "Needle_Passing", "SonicDrawing", "Suturing"]
    seq_dirs = [os.path.join(dataset, seq) for seq in sequences]
    output_dir = "frames"
    extract_frames(seq_dirs, output_dir)
