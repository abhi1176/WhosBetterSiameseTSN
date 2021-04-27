import cv2
import os

from argparse import ArgumentParser
from glob import glob

def main(seq):
	dataset_dir = os.path.join("dataset", seq)
	frames_dir = os.path.join("frames", seq)

	for idx, dir_ in enumerate([dir_ for dir_ in os.listdir(dataset_dir)]):
		video = os.path.join(dataset_dir, dir_)
		cap = cv2.VideoCapture(video)
		fname, _ = os.path.splitext(dir_)
		frames = os.path.join(frames_dir, fname)
		rgb_frames = flow_frames = 0
		try:
			rgb_frames = len(glob(os.path.join(frames, "rgb_*")))
			flow_frames = len(glob(os.path.join(frames, "flow_*")))
		except:
			pass
		print(idx, video, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), rgb_frames, flow_frames)

parser = ArgumentParser()
parser.add_argument("-s", "--sequences", nargs="+", required=True)
args = parser.parse_args()

for seq in args.sequences:
	main(seq)
