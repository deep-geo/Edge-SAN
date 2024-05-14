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
import cv2
import numpy as np
import tqdm
import random

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
            label = labels[i, :, :, 0]  # 0 - instance, 1 - semantic
            vals = [_ for _ in np.unique(label) if _ != 0]
            random.shuffle(vals)
            val_groups = [vals[i:i + 255] for i in range(0, len(vals), 255)]
            for j, group in enumerate(val_groups):
                step = self.calc_step(len(vals))
                dst_label = np.zeros(shape=(label.shape[0], label.shape[1]), dtype=np.uint8)
                for k, val in enumerate(group):
                    tmp_label = label.copy().astype(np.uint8)
                    tmp_label[label != val] = 0
                    tmp_label[label == val] = 255
                    tmp_contours, _ = cv2.findContours(tmp_label,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    dst_label = cv2.drawContours(dst_label, tmp_contours, -1,
                                                 color=255 - k * step, thickness=-1)

                dst_label = self.transform(dst_label)

                if j == 0:
                    label_name = self.dst_prefix + f"image_{i + 1:04d}.png"
                else:
                    label_name = self.dst_prefix + f"image_{i + 1:04d}({j}).png"

                dst_path = os.path.join(self.dst_label_dir, label_name)
                cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessCoNIC(args.src_root, args.dst_root,
                    args.dst_size, args.dst_prefix).process()
