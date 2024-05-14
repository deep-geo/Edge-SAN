"""
Script for lizard dataset preprocessing.

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
│       ├── consep_15.png
│       ...
└── train
    ├── bound
    │   ├── consep_10.png
    │   ├── consep_12.png
    │   ├── consep_13.png
    │   ...
    ├── data
    │   ├── consep_10.png
    │   ├── consep_12.png
    │   ├── consep_13.png
    │   ...
    └── label
        ├── consep_10.png
        ├── consep_12.png
        ├── consep_13.png
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


class PreprocessLizard(Preprocess):

    def process(self):

        for split in ["test", "train"]:
            # data
            src_data_dir = os.path.join(self.src_root, split, "data")
            src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
            print(f"\nProcess {split} data...")
            for src_data_path in tqdm.tqdm(src_data_paths):
                img = cv2.imread(src_data_path)
                dst_img = self.transform(img)
                dst_data_path = os.path.join(
                    self.dst_data_dir,
                    self.dst_prefix + os.path.basename(src_data_path)
                )
                cv2.imwrite(dst_data_path, dst_img)

            # label
            src_label_dir = os.path.join(self.src_root, split, "label")    # bound, not label
            label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
            print(f"\nProcess {split} label...")
            for path in tqdm.tqdm(label_paths):
                basename = os.path.basename(path)
                label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                _, label = cv2.threshold(label, 200, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = np.ones((3, 3), np.uint8)
                sure_bg = cv2.dilate(label, kernel, iterations=3)
                dist_transform = cv2.distanceTransform(label, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0, 255, 0)
                sure_fg = np.uint8(sure_fg)
                _, markers = cv2.connectedComponents(sure_fg)

                vals = [val for val in np.unique(markers) if val != 0]
                random.shuffle(vals)

                val_groups = [vals[i:i + 255] for i in range(0, len(vals), 255)]
                for i, group in enumerate(val_groups):
                    step = self.calc_step(len(group))
                    dst_label = np.zeros(shape=label.shape, dtype=np.uint8)
                    for j, val in enumerate(group):
                        dst_label[markers == val] = 255 - j * step

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

    PreprocessLizard(args.src_root, args.dst_root,
                     args.dst_size, args.dst_prefix).process()
