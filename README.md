# Download & Prepare datasets
```sh
git clone https://github.com/hazeld/EPIC-Skills2018.git

mkdir dataset && cd dataset

# [ChopstickUsing, HandDrawing and SonicDrawing]
# Download https://drive.google.com/file/d/1Ck-Dke5AcKMzKsedJfblRcTjSPBklo4P/view to `dataset`
unzip EPIC-Skills2018_videos.zip  # This didn't work on Linux, but did on Windows.
cd EPIC-Skills2018_videos/videos
for dir in `ls`
do
        cd $dir && rm -rf Headmounted && mv Stationary/* . && rm -rf Stationary && cd ../
done
cd ../../
mv EPIC-Skills2018_videos/videos/* .
rm -rf EPIC-Skills2018_videos


# [DoughRolling]
mkdir DoughRolling_zip DoughRolling
# Download all video.zip files in http://kitchen.cs.cmu.edu/main.php?recipefilter=Pizza#table to DoughRolling_zip
cd DoughRolling_zip
for i in S07 S08 S09 S11 S12 S14 S15 S16 S17 S18 S19 S20 S22 S25 S28 S29 S30 S31 S32 S33 S34 S35 S36 S37 S40 S41 S47 S48 S49 S50 S51 S52 S53 S54 S55
do
        wget http://kitchen.cs.cmu.edu/Main/${i}_Pizza_Video.zip
        unzip ${i}_Pizza_Video.zip -d ${i}_Pizza_Video
done
files=`cat ../../EPIC-Skills2018/dough_rolling_segments.csv | awk -F "," '{ print $1 }' | xargs | sed "s/^[^ ]* //"`
for file in ${files}
do
        mv */*${file}* ../DoughRolling
done
rm -rf DoughRolling_zip
cd ../


# [Suturing, NeedlePassing and KnotTying]
# Follow and instructions and download dataset from https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
for file in Needle_Passing Knot_Tying Suturing
do
unzip ${file}.zip && \
        cd ${file} && \
        rm -r !("video") && \  # Remove all files and folders except `video` directory. Do not run this as a sudo user.
        mv video/*_capture2.avi . && rm -rf video && cd ../  # Move all video files from video directory
done
```


# Setup

## Create conda environment
```sh
conda create -n ai python=3.8
conda activate ai
```

## Install packages
```sh
conda install tensorflow-gpu
pip install opencv-python
conda install numpy pandas scikit-image pydot
```


# Prepare Dataset

```sh
python prepare_data.py -w 10 -s ChopstickUsing HandDrawing SonicDrawing DoughRolling Suturing Needle_Passing Knot_Tying
```


# Prepare Annotations

```sh
python rename_frames_dir.py

python find_similar_pairs.py -s Chopstick_Using
python find_similar_pairs.py -s Hand_Drawing
python find_similar_pairs.py -s Sonic_Drawing
python find_similar_pairs.py -s Dough_Rolling
python find_similar_pairs.py -s Suturing
python find_similar_pairs.py -s Needle_Passing
python find_similar_pairs.py -s Knot_Tying

python prepare_annotations.py
```

# Get pretrained weights

```sh
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

# Train Spatial Model

```sh
python train_spatial.py -b 128 -s 7
python train_spatial.py -b 64 -s 21
```

# Train Temporal Model

```sh
python train_temporal.py -b 128 -s 1
python train_temporal.py -b 128 -s 3
python train_temporal.py -b 64 -s 7
```

# Test

## ChopstickUsing
```sh
python evaluate.py \
    -i val_ChopstickUsing.csv \
    -sm spatial_models_7/spatial_model_iter_0570.h5 \
    -tm temporal_models_1/temporal_model_iter_610.h5
# Snippets: 7
# Accurarcy: 0.979
# Spatial Accuracy: 0.974
# Temporal Accuracy: 0.864
```

## DoughRolling
```sh
python evaluate.py \
    -i val_DoughRolling.csv \
    -sm spatial_models_7/spatial_model_iter_0570.h5 \
    -tm temporal_models_1/temporal_model_iter_610.h5
# Snippets: 7
# Accurarcy: 0.891
# Spatial Accuracy: 0.922
# Temporal Accuracy: 0.578
```

## HandDrawing
```sh
python evaluate.py -b 59\
    -i val_HandDrawing.csv \
    -sm spatial_models_7/spatial_model_iter_0570.h5 \
    -tm temporal_models_1/temporal_model_iter_610.h5
# Snippets: 7
# Accurarcy: 0.949
# Spatial Accuracy: 0.949
# Temporal Accuracy: 0.746
```

## KnotTying
```sh
python evaluate.py \
    -i val_KnotTying.csv \
    -sm spatial_models_7/spatial_model_iter_0570.h5 \
    -tm temporal_models_1/temporal_model_iter_610.h5
# Snippets: 7
# Accurarcy: 0.930
# Spatial Accuracy: 0.926
# Temporal Accuracy: 0.648
```

## NeedlePassing
```sh
python evaluate.py -b 80 \
    -i val_NeedlePassing.csv \
    -sm spatial_models_7/spatial_model_iter_0570.h5 \
    -tm temporal_models_1/temporal_model_iter_610.h5
# Snippets: 7
# Accurarcy: 0.938
# Spatial Accuracy: 0.914
# Temporal Accuracy: 0.727
```

## SonicDrawing
```sh
python evaluate.py -b 47 \
    -i val_SonicDrawing.csv \
    -sm spatial_models_7/spatial_model_iter_0570.h5 \
    -tm temporal_models_1/temporal_model_iter_610.h5
# Snippets: 7
# Accurarcy: 0.872
# Spatial Accuracy: 0.915
# Temporal Accuracy: 0.638
```

## Suturing
```sh
python evaluate.py -b 100 \
    -i val_Suturing.csv \
    -sm spatial_models_7/spatial_model_iter_0570.h5 \
    -tm temporal_models_1/temporal_model_iter_610.h5
# Snippets: 7
# Accurarcy: 0.880
# Spatial Accuracy: 0.893
# Temporal Accuracy: 0.650
```
