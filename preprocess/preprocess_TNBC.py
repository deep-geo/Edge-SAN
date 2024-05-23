"""
Script for TNBC dataset preprocessing.

Original dataset structure:
.
├── Images
│   ├── 5784
│   │   ├── 01_1.png
│   │   ├── 01_2.png
│   │   ├── 01_3.png
│   │   ...
│   ├── A13E
│   │   ├── 01_1.png
│   │   ├── 01_2.png
│   │   ├── 01_3.png
│   │   ...
│   ├── A1AS
│   │   ├── 01_1.png
│   │   ├── 01_2.png
│   │   ├── 01_3.png
│   │   ...
│   ├── A2I6
│   │   ├── 01_1.png
│   │   ├── 01_2.png
│   │   ├── 01_3.png
│   │   ...
│   └── XXXX
│       ├── 01_1.png
│       ├── 01_2.png
│       ├── 01_3.png
│       ...
├── Labels
│   ├── 01_1.mat
│   ├── 01_2.mat
│   ├── 01_3.mat
│   ...
└── Overlay
    ├── 141549_283.png
    ├── 141549_367.png
    ├── 141549_83.png
    ...
"""
import os
import glob
import argparse
import cv2
import tqdm

from preprocess import Preprocess
from scipy.io import loadmat


class PreprocessTNBC(Preprocess):

    def process(self):
        # data
        src_data_dir = os.path.join(self.src_root, "Images", "5784")
        src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
        print("\nProcess data...")
        for path in tqdm.tqdm(src_data_paths):
            img = cv2.imread(path)
            self.save_data(ori_data=img, data_name=os.path.basename(path)[:-4])

        # label
        src_label_dir = os.path.join(self.src_root, "Labels")
        label_paths = glob.glob(os.path.join(src_label_dir, "*.mat"))
        print("\nProcess label...")
        for path in tqdm.tqdm(label_paths):
            label = loadmat(path)["inst_map"]
            self.save_label(ori_label=label,
                            label_name=os.path.basename(path)[:-4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessTNBC(args.src_root, args.dst_root,
                   args.dst_size, args.dst_prefix).process()
