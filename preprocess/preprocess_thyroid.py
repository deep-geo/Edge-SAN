"""
Script for thyroid dataset preprocessing.

Original dataset structure:
.
├── test
│   ├── bound
│   │   ├── consep_1.png
│   │   ├── consep_11.png
│   │   ├── consep_14.png
│   │   ...
│   ├── data
│   │   ├── consep_1.png
│   │   ├── consep_11.png
│   │   ├── consep_14.png
│   │   ...
│   └── label
│       ├── consep_1.png
│       ├── consep_11.png
│       ├── consep_14.png
│       ...
└── train
    ├── bound
    │   ├── consep_2.png
    │   ├── consep_3.png
    │   ├── consep_4.png
    │   ...
    ├── data
    │   ├── consep_2.png
    │   ├── consep_3.png
    │   ├── consep_4.png
    │   ...
    └── label
        ├── consep_2.png
        ├── consep_3.png
        ├── consep_4.png
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


class PreprocessThyroid(Preprocess):

    def process(self):

        for split in ["test", "train"]:
            # data
            src_data_dir = os.path.join(self.src_root, split, "data")
            src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
            print(f"\nProcess original {split} data...")
            for path in tqdm.tqdm(src_data_paths):
                img = cv2.imread(path)
                self.save_data(ori_data=img,
                               data_name=os.path.basename(path)[:-4])

            # label
            src_label_dir = os.path.join(self.src_root, split, "label")    # bound, not label
            src_bound_dir = os.path.join(self.src_root, split, "bound")    # bound, not label
            label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
            print(f"\nProcess original {split} label...")
            for path in tqdm.tqdm(label_paths):
                basename = os.path.basename(path)
                src_label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                bound_path = os.path.join(src_bound_dir, basename)
                src_bound = cv2.imread(bound_path, cv2.IMREAD_GRAYSCALE)
                src_bound[src_bound != 2] = 0
                src_label[src_bound == 2] = 0   # 去掉相邻边界

                contours, _ = cv2.findContours(src_label, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                contours = list(contours)
                random.shuffle(contours)
                label_uint16 = np.zeros(shape=src_label.shape, dtype=np.uint16)
                for i, contour in enumerate(contours):
                    tmp_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
                    tmp_label = cv2.drawContours(tmp_label, [contour], -1, 255, -1)

                    # 膨胀补上边缘区域
                    dilation_kernel = np.ones((3, 3), np.uint8)
                    tmp_label = cv2.dilate(tmp_label, dilation_kernel, iterations=1)

                    tmp_contours, _ = cv2.findContours(tmp_label,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    label_uint16 = cv2.drawContours(
                        label_uint16, tmp_contours, -1, color=i + 1, thickness=-1
                    )

                self.save_label(ori_label=label_uint16, label_name=basename[:-4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessThyroid(args.src_root, args.dst_root,
                      args.dst_size, args.dst_prefix).process()
