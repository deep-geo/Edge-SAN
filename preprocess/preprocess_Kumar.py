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
import tqdm

from preprocess import Preprocess
from scipy.io import loadmat


class PreprocessKumar(Preprocess):

    def process(self):
        for split in ["test_diff", "test_same", "train"]:
            # data
            src_data_dir = os.path.join(self.src_root, split, "Images")
            src_data_paths = glob.glob(os.path.join(src_data_dir, "*.tif"))
            print(f"\nProcess {split} data...")
            for path in tqdm.tqdm(src_data_paths):
                img = cv2.imread(path)
                self.save_data(ori_data=img,
                               data_name=os.path.basename(path)[:-4])

            # label
            src_label_dir = os.path.join(self.src_root, split, "Labels")
            label_paths = glob.glob(os.path.join(src_label_dir, "*.mat"))
            print(f"\nProcess {split} label...")
            for path in tqdm.tqdm(label_paths):
                label = loadmat(path)["inst_map"]
                self.save_label(ori_label=label,
                                label_name=os.path.basename(path)[:-4])

        self.save_info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessKumar(args.src_root, args.dst_root,
                    args.dst_size, args.dst_prefix).process()
