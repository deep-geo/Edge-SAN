"""
Script for fluorescence dataset preprocessing.

Original dataset structure:
.
├── data
│   ├── image0.png
│   ├── image1.png
│   ├── image2.png
│   ...
└── label
    ├── image0.png
    ├── image1.png
    ├── image2.png
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


class PreprocessFluorescence(Preprocess):

    def process(self):
        # data
        src_data_dir = os.path.join(self.src_root, "data")
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
        src_label_dir = os.path.join(self.src_root, "label")
        label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
        print("\nProcess label...")
        for path in tqdm.tqdm(label_paths):
            src_label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            vals = [_ for _ in np.unique(src_label) if _ != 0]
            step = self.calc_step(len(vals))
            dst_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
            random.shuffle(vals)
            for i, val in enumerate(vals):
                dst_label[src_label == val] = 255 - i * step

            dst_label = self.transform(dst_label)

            label_name = os.path.basename(path)
            dst_path = os.path.join(self.dst_label_dir,
                                    self.dst_prefix + label_name)
            cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessFluorescence(args.src_root, args.dst_root,
                           args.dst_size, args.dst_prefix).process()
