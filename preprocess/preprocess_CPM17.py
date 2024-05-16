"""
Script for CPM17 dataset preprocessing.

Original dataset structure:
.
├── test
│   ├── Images
│   │   ├── image_00.png
│   │   ├── image_01.png
│   │   ├── image_02.png
│   │   ...
│   ├── Labels
│   │   ├── image_00.mat
│   │   ├── image_01.mat
│   │   ├── image_02.mat
│   │   ...
│   └── Overlay
│       ├── image_00.png
│       ├── image_01.png
│       ├── image_02.png
│       ...
└── train
    ├── Images
    │   ├── image_00.png
    │   ├── image_01.png
    │   ├── image_02.png
    │   ...
    ├── Labels
    │   ├── image_00.mat
    │   ├── image_01.mat
    │   ├── image_02.mat
    │   ...
    └── Overlay
        ├── image_00.png
        ├── image_01.png
        ├── image_02.png
        ...
"""
import os
import glob
import argparse
import cv2
import numpy as np
import tqdm
import random

from preprocess import Preprocess
from scipy.io import loadmat
from preprocess_CPM15 import PreprocessCPM15


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    for split in ["test", "train"]:
        PreprocessCPM15(
            os.path.join(args.src_root, split),
            args.dst_root,
            args.dst_size,
            args.dst_prefix
        ).process()
