"""
Script for Kumar dataset preprocessing.

Original dataset structure:
.
├── test_diff
│   ├── Images
│   │   ├── TCGA-AY-A8YK-01A-01-TS1.tif
│   │   ├── TCGA-DK-A2I6-01A-01-TS1.tif
│   │   ├── TCGA-G2-A2EK-01A-02-TSB.tif
│   │   ...
│   ├── Labels
│   │   ├── TCGA-AY-A8YK-01A-01-TS1.mat
│   │   ├── TCGA-DK-A2I6-01A-01-TS1.mat
│   │   ├── TCGA-G2-A2EK-01A-02-TSB.mat
│   │   ...
│   └── Overlay
│       ├── TCGA-AY-A8YK-01A-01-TS1.png
│       ├── TCGA-DK-A2I6-01A-01-TS1.png
│       ├── TCGA-G2-A2EK-01A-02-TSB.png
│       ...
├── test_same
│   ├── Images
│   │   ├── TCGA-21-5784-01Z-00-DX1.tif
│   │   ├── TCGA-21-5786-01Z-00-DX1.tif
│   │   ├── TCGA-B0-5698-01Z-00-DX1.tif
│   │   ...
│   ├── Labels
│   │   ├── TCGA-21-5784-01Z-00-DX1.mat
│   │   ├── TCGA-21-5786-01Z-00-DX1.mat
│   │   ├── TCGA-B0-5698-01Z-00-DX1.mat
│   │   ...
│   └── Overlay
│       ├── TCGA-21-5784-01Z-00-DX1.png
│       ├── TCGA-21-5786-01Z-00-DX1.png
│       ├── TCGA-B0-5698-01Z-00-DX1.png
│       ...
└── train
    ├── Images
    │   ├── TCGA-18-5592-01Z-00-DX1.tif
    │   ├── TCGA-38-6178-01Z-00-DX1.tif
    │   ├── TCGA-49-4488-01Z-00-DX1.tif
    │   ...
    ├── Labels
    │   ├── TCGA-18-5592-01Z-00-DX1.mat
    │   ├── TCGA-38-6178-01Z-00-DX1.mat
    │   ├── TCGA-49-4488-01Z-00-DX1.mat
    │   ...
    └── Overlay
        ├── TCGA-18-5592-01Z-00-DX1.png
        ├── TCGA-38-6178-01Z-00-DX1.png
        ├── TCGA-49-4488-01Z-00-DX1.png
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


class PreprocessKumar(Preprocess):

    def process(self):
        for split in ["test_diff", "test_same", "train"]:
            # data
            src_data_dir = os.path.join(self.src_root, split, "Images")
            src_data_paths = glob.glob(os.path.join(src_data_dir, "*.tif"))

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
            src_label_dir = os.path.join(self.src_root, split, "Labels")
            label_paths = glob.glob(os.path.join(src_label_dir, "*.mat"))
            print(f"\nProcess {split} label...")
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
                        dst_label_uint8[
                            dst_label_uint16 == val] = 255 - j * step

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

    PreprocessKumar(args.src_root, args.dst_root,
                    args.dst_size, args.dst_prefix).process()
