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
import numpy as np
import tqdm
import random

from preprocess import Preprocess
from scipy.io import loadmat


class PreprocessCoNSeP(Preprocess):

    def process(self):
        for split in ["Test", "Train"]:
            # data
            src_data_dir = os.path.join(self.src_root, split, "Images")
            src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))

            print("\nProcess data...")
            for src_data_path in tqdm.tqdm(src_data_paths):
                img = cv2.imread(src_data_path)
                dst_img = self.transform(img)
                dst_data_path = os.path.join(
                    self.dst_data_dir,
                    self.dst_prefix + os.path.basename(src_data_path)
                )
                cv2.imwrite(dst_data_path, dst_img)

            # label
            src_label_dir = os.path.join(self.src_root, split, "Labels")
            label_paths = glob.glob(os.path.join(src_label_dir, "*.mat"))
            print("\nProcess label...")
            for path in tqdm.tqdm(label_paths):
                label = loadmat(path)["inst_map"].astype(np.uint16)
                dst_label_uint16 = self.transform(label)

                basename = os.path.basename(path)[:-4]

                # npy
                dst_arr_path = os.path.join(self.dst_label_dir,
                                            self.dst_prefix + f"{basename}.npy")
                np.save(dst_arr_path, dst_label_uint16)

                # png
                vals_uint16 = [_ for _ in np.unique(dst_label_uint16) if _ != 0][:255]
                dst_label_uint8 = np.zeros(shape=dst_label_uint16.shape,
                                           dtype=np.uint8)
                if len(vals_uint16) > 0:
                    random.shuffle(vals_uint16)
                    step = self.calc_step(len(vals_uint16))
                    for j, val in enumerate(vals_uint16):
                        dst_label_uint8[dst_label_uint16 == val] = 255 - j * step

                dst_img_path = os.path.join(self.dst_label_dir,
                                            self.dst_prefix + f"{basename}.png")
                cv2.imwrite(dst_img_path, dst_label_uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessCoNSeP(args.src_root, args.dst_root,
                     args.dst_size, args.dst_prefix).process()
