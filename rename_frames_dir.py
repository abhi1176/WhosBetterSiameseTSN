
import os

frames = "frames"

for dir_ in os.listdir(frames):
	dir_path = os.path.join(frames, dir_)
	for seq in os.listdir(dir_path):
		if dir_ == "Suturing":
			new_seq = seq.replace("Suturing_", "").replace("_capture2", "")
		elif dir_ == "Needle_Passing":
			new_seq = seq.replace("Needle_Passing_", "").replace("_capture2", "")
		elif dir_ == "Knot_Tying":
			new_seq = seq.replace("Knot_Tying_", "").replace("_capture2", "")
		os.rename(os.path.join(dir_path, seq), os.path.join(dir_path, new_seq))
