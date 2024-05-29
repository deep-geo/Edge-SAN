"""
Script for CoNSeP dataset preprocessing.

Original dataset structure:
.
├── README.txt
├── Test
│   ├── Images
│   │   ├── test_1.png
│   │   ├── test_10.png
│   │   ├── test_11.png
│   │   ...
│   ├── Labels
│   │   ├── test_1.mat
│   │   ├── test_10.mat
│   │   ├── test_11.mat
│   │   ...
│   └── Overlay
│       ├── test_1.png
│       ├── test_10.png
│       ├── test_11.png
│       ...
└── Train
    ├── Images
    │   ├── train_1.png
    │   ├── train_10.png
    │   ├── train_11.png
    │   ...
    ├── Labels
    │   ├── train_1.mat
    │   ├── train_10.mat
    │   ├── train_11.mat
    │   ...
    └── Overlay
        ├── train_1.png
        ├── train_10.png
        ├── train_11.png
        ...
"""
import os
import glob
import argparse
import cv2
import tqdm

from preprocess import Preprocess
from scipy.io import loadmat


class PreprocessCoNSeP(Preprocess):

    def process(self):
        for split in ["Test", "Train"]:
            # data
            src_data_dir = os.path.join(self.src_root, split, "Images")
            src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
            print("\nProcess data...")
            for path in tqdm.tqdm(src_data_paths):
                img = cv2.imread(path)
                self.save_data(ori_data=img,
                               data_name=os.path.basename(path)[:-4])

            # label
            src_label_dir = os.path.join(self.src_root, split, "Labels")
            label_paths = glob.glob(os.path.join(src_label_dir, "*.mat"))
            print("\nProcess label...")
            for path in tqdm.tqdm(label_paths):
                label = loadmat(path)["inst_map"]
                self.save_label(ori_label=label,
                                label_name=os.path.basename(path)[:-4])

        self.save_info(info_data={"count": self.count})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessCoNSeP(args.src_root, args.dst_root,
                     args.dst_size, args.dst_prefix).process()
