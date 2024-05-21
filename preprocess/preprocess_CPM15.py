"""
Script for CPM15 dataset preprocessing.

Original dataset structure:
.
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
import tqdm

from preprocess import Preprocess
from scipy.io import loadmat


class PreprocessCPM15(Preprocess):

    def process(self):
        # data
        src_data_dir = os.path.join(self.src_root, "Images")
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
            self.save_label(ori_label=label, label_name=os.path.basename(path)[:-4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessCPM15(args.src_root, args.dst_root,
                    args.dst_size, args.dst_prefix).process()
