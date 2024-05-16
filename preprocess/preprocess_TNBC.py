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
import numpy as np
import tqdm
import random

from preprocess import Preprocess
from scipy.io import loadmat


class PreprocessTNBC(Preprocess):

    def process(self):
        # data
        src_data_dir = os.path.join(self.src_root, "Images", "5784")
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
        src_label_dir = os.path.join(self.src_root, "Labels")
        label_paths = glob.glob(os.path.join(src_label_dir, "*.mat"))
        print("\nProcess label...")
        for path in tqdm.tqdm(label_paths):
            label = loadmat(path)["inst_map"]
            vals = [_ for _ in np.unique(label) if _ != 0]
            random.shuffle(vals)
            val_groups = [vals[i:i + 255] for i in range(0, len(vals), 255)]
            for i, group in enumerate(val_groups):
                step = self.calc_step(len(group))
                dst_label = np.zeros(
                    shape=(label.shape[0], label.shape[1]),
                    dtype=np.uint8
                )
                for k, val in enumerate(group):
                    tmp_label = label.copy().astype(np.uint8)
                    tmp_label[label != val] = 0
                    tmp_label[label == val] = 255
                    tmp_contours, _ = cv2.findContours(tmp_label,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    dst_label = cv2.drawContours(dst_label, tmp_contours,
                                                 -1,
                                                 color=255 - k * step,
                                                 thickness=-1)

                dst_label = self.transform(dst_label)

                basename = os.path.basename(path)[:-4] + ".png"

                if i == 0:
                    label_name = basename
                else:
                    label_name = f"{basename[:-4]}({i}).png"

                dst_path = os.path.join(self.dst_label_dir, label_name)
                cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessTNBC(args.src_root, args.dst_root,
                   args.dst_size, args.dst_prefix).process()
