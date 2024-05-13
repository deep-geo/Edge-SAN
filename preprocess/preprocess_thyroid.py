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

        for sub_dir in ["test", "train"]:
            # data
            src_data_dir = os.path.join(self.src_root, sub_dir, "data")
            src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
            print(f"\nProcess original {sub_dir} data...")
            for src_data_path in tqdm.tqdm(src_data_paths):
                img = cv2.imread(src_data_path)
                dst_img = self.transform(img)
                dst_data_path = os.path.join(
                    self.dst_data_dir,
                    self.dst_prefix + os.path.basename(src_data_path)
                )
                cv2.imwrite(dst_data_path, dst_img)

            # label
            src_label_dir = os.path.join(self.src_root, sub_dir, "label")    # bound, not label
            src_bound_dir = os.path.join(self.src_root, sub_dir, "bound")    # bound, not label
            label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
            print(f"\nProcess original {sub_dir} label...")
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

                contour_groups = [contours[i:i + 255] for i in range(0, len(contours), 255)]
                for i, group in enumerate(contour_groups):
                    step = self.calc_step(len(group))
                    dst_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
                    for j, contour in enumerate(group):
                        tmp_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
                        tmp_label = cv2.drawContours(tmp_label, [contour], -1, 255, -1)

                        # 膨胀补上边缘区域
                        dilation_kernel = np.ones((3, 3), np.uint8)
                        tmp_label = cv2.dilate(tmp_label, dilation_kernel, iterations=1)

                        tmp_contours, _ = cv2.findContours(tmp_label,
                                                           cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                        dst_label = cv2.drawContours(dst_label, tmp_contours, -1,
                                                     color=255 - j * step, thickness=-1)

                    dst_label = self.transform(dst_label)

                    if i == 0:
                        label_name = self.dst_prefix + basename
                    else:
                        label_name = self.dst_prefix + f"{basename[:-4]}({i}).png"

                    dst_path = os.path.join(self.dst_label_dir, label_name)
                    cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessThyroid(args.src_root, args.dst_root,
                      args.dst_size, args.dst_prefix).process()

