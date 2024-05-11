"""
Script for thyroid dataset preprocessing.

Original dataset structure:
| - CoNIC/
    | - images.npy
    | - labels.npy
    | - patch_info.csv
    | - README.txt

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
    else:
        os.makedirs(dst_root)

    images_path = os.path.join(src_root, "images.npy")
    labels_path = os.path.join(src_root, "labels.npy")

    dst_data_dir = os.path.join(dst_root, "data")
    dst_label_dir = os.path.join(dst_root, "label")
    os.makedirs(dst_data_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    images = np.load(images_path)
    transform = get_transform(image_size, images.shape[1], images.shape[2])

    # data
    print("\nProcess data...")
    for i in tqdm.tqdm(range(images.shape[0]), total=images.shape[0]):
        img = transform(image=images[i])["image"]
        dst_img_path = os.path.join(dst_data_dir, f"image_{i + 1:04d}.png")
        cv2.imwrite(dst_img_path, img)

    # label
    labels = np.load(labels_path)
    print("\nProcess label...")
    for i in tqdm.tqdm(range(labels.shape[0]), total=labels.shape[0]):
        label = labels[i, :, :, 0]  # 0 - instance, 1 - semantic
        vals = [_ for _ in np.unique(label) if _ != 0]
        random.shuffle(vals)
        val_groups = [vals[i:i + 255] for i in range(0, len(vals), 255)]
        for j, group in enumerate(val_groups):
            step = max(1, min(STEP, int((255 - 25) / len(vals))))
            dst_label = np.zeros(shape=(label.shape[0], label.shape[1]), dtype=np.uint8)
            for k, val in enumerate(group):
                tmp_label = label.copy().astype(np.uint8)
                tmp_label[tmp_label != val] = 0
                tmp_label[tmp_label == val] = 255
                tmp_contours, _ = cv2.findContours(tmp_label,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                dst_label = cv2.drawContours(dst_label, tmp_contours, -1,
                                             color=255 - k * step, thickness=-1)

            dst_label = transform(image=dst_label)["image"]

            if j == 0:
                label_name = f"image_{i + 1:04d}.png"
            else:
                label_name = f"image_{i + 1:04d}({j}).png"

            dst_path = os.path.join(dst_label_dir, label_name)
            cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--image_size", type=int)
    args = parser.parse_args()

    preprocess(args.src_root, args.dst_root, args.image_size)
