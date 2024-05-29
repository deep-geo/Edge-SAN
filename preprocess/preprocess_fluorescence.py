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
import tqdm

from preprocess import Preprocess


class PreprocessFluorescence(Preprocess):

    def process(self):
        # data
        src_data_dir = os.path.join(self.src_root, "data")
        src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
        print("\nProcess data...")
        for path in tqdm.tqdm(src_data_paths):
            img = cv2.imread(path)
            self.save_data(ori_data=img, data_name=os.path.basename(path)[:-4])

        # label
        src_label_dir = os.path.join(self.src_root, "label")
        label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
        print("\nProcess label...")
        for path in tqdm.tqdm(label_paths):
            label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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

    PreprocessFluorescence(args.src_root, args.dst_root,
                           args.dst_size, args.dst_prefix).process()
