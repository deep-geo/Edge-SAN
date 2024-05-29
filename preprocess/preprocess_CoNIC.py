"""
Script for CoNIC dataset preprocessing.

Original dataset structure:
.
├── README.txt
├── images.npy
├── labels.npy
└── patch_info.csv

"""
import os
import argparse
import numpy as np
import tqdm

from preprocess import Preprocess


class PreprocessCoNIC(Preprocess):

    def process(self):

        images_path = os.path.join(self.src_root, "images.npy")
        labels_path = os.path.join(self.src_root, "labels.npy")

        # data
        images = np.load(images_path)
        print("\nProcess data...")
        for i in tqdm.tqdm(range(images.shape[0]), total=images.shape[0]):
            self.save_data(ori_data=images[i], data_name=f"image_{i + 1:04d}")

        # label
        labels = np.load(labels_path)
        print("\nProcess label...")
        for i in tqdm.tqdm(range(labels.shape[0]), total=labels.shape[0]):
            label = labels[i, :, :, 0]  # 0 - instance, 1 - semantic
            self.save_label(ori_label=label, label_name=f"image_{i + 1:04d}")

        self.save_info(info_data={"count": self.count})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessCoNIC(args.src_root, args.dst_root,
                    args.dst_size, args.dst_prefix).process()
