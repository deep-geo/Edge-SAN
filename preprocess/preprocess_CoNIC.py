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
import random

import cv2
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
            dst_img = self.transform(images[i])
            dst_img_path = os.path.join(
                self.dst_data_dir,
                self.dst_prefix + f"image_{i + 1:04d}.png"
            )
            cv2.imwrite(dst_img_path, dst_img)

        # label
        labels = np.load(labels_path)
        print("\nProcess label...")
        for i in tqdm.tqdm(range(labels.shape[0]), total=labels.shape[0]):
            label = labels[i, :, :, 0].astype(np.uint16)  # 0 - instance, 1 - semantic
            dst_label_uint16 = self.transform(label)

            # npy
            dst_arr_path = os.path.join(self.dst_label_dir,
                                        self.dst_prefix + f"image_{i + 1:04d}.npy")
            np.save(dst_arr_path, dst_label_uint16)

            # png
            vals_uint16 = [_ for _ in np.unique(dst_label_uint16) if _ != 0][:255]
            dst_label_uint8 = np.zeros(shape=dst_label_uint16.shape, dtype=np.uint8)
            if len(vals_uint16) > 0:
                random.shuffle(vals_uint16)
                step = self.calc_step(len(vals_uint16))
                for j, val in enumerate(vals_uint16):
                    dst_label_uint8[dst_label_uint16 == val] = 255 - j * step

            dst_img_path = os.path.join(self.dst_label_dir,
                                        self.dst_prefix + f"image_{i + 1:04d}.png")
            cv2.imwrite(dst_img_path, dst_label_uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessCoNIC(args.src_root, args.dst_root,
                    args.dst_size, args.dst_prefix).process()
