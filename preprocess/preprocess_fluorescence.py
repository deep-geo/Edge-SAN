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
import numpy as np
import tqdm
import random

from utils import get_transform


STEP = 20


def preprocess(src_root: str, dst_root: str, image_size: int):
    if os.path.exists(dst_root):
        assert not glob.glob(os.path.join(dst_root, "*")), f"dst_root is supposed to be empty: {dst_root}"

    # data - just copy
    src_data_dir = os.path.join(src_root, "data")
    src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
    dst_data_dir = os.path.join(dst_root, "data")
    os.makedirs(dst_data_dir, exist_ok=True)
    print("\nProcess data...")
    for src_data_path in tqdm.tqdm(src_data_paths):
        img = cv2.imread(src_data_path)
        transform = get_transform(image_size, img.shape[0], img.shape[1])
        img = transform(image=img)["image"]
        dst_data_path = os.path.join(dst_data_dir, os.path.basename(src_data_path))
        cv2.imwrite(dst_data_path, img)

    # label
    src_label_dir = os.path.join(src_root, "label")
    dst_label_dir = os.path.join(dst_root, "label")
    os.makedirs(dst_label_dir, exist_ok=True)
    label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
    print("\nProcess label...")
    for path in tqdm.tqdm(label_paths):
        src_label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        vals = [_ for _ in np.unique(src_label) if _ != 0]
        step = max(1, min(STEP, int((255 - 25) / len(vals))))
        dst_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
        random.shuffle(vals)
        for i, val in enumerate(vals):
            dst_label[src_label == val] = 255 - i * step

        label_name = os.path.basename(path)
        dst_path = os.path.join(dst_label_dir, label_name)
        transform = get_transform(image_size, dst_label.shape[0], dst_label.shape[1])
        dst_label = transform(image=dst_label)["image"]
        cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--image_size", type=int)
    args = parser.parse_args()

    preprocess(args.src_root, args.dst_root, args.image_size)
