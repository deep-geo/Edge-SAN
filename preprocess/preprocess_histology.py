"""
Script for histology dataset preprocessing.

Original dataset structure:
.
├── data
│   ├── 0TCGA-21-5784-01Z-00-DX1.png
│   ├── 0TCGA-21-5786-01Z-00-DX1.png
│   ├── 0TCGA-2Z-A9J9-01A-01-TS1.png
│   ...
└── label
    ├── 0TCGA-21-5784-01Z-00-DX1.png
    ├── 0TCGA-21-5786-01Z-00-DX1.png
    ├── 0TCGA-2Z-A9J9-01A-01-TS1.png
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
    else:
        os.makedirs(dst_root)

    # data
    src_data_dir = os.path.join(src_root, "data")
    src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
    dst_data_dir = os.path.join(dst_root, "data")
    os.makedirs(dst_data_dir, exist_ok=True)
    print("\nProcess data...")
    for src_data_path in tqdm.tqdm(src_data_paths):
        img = cv2.imread(src_data_path)
        transform = get_transform(image_size, img.shape[0], img.shape[1])
        img = transform(image=img)["image"]
        dst_data_path = os.path.join(dst_data_dir,
                                     os.path.basename(src_data_path))
        cv2.imwrite(dst_data_path, img)

    # label
    src_label_dir = os.path.join(src_root, "label")    # bound, not label
    dst_label_dir = os.path.join(dst_root, "label")
    os.makedirs(dst_label_dir, exist_ok=True)
    label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
    print("\nProcess label...")
    for path in tqdm.tqdm(label_paths):
        basename = os.path.basename(path)
        src_label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # remove bound
        src_label[src_label == 76] = 0
        src_label[src_label == 149] = 0

        contours, _ = cv2.findContours(src_label, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        random.shuffle(contours)

        contour_groups = [contours[i:i + 255] for i in range(0, len(contours), 255)]
        for i, group in enumerate(contour_groups):
            step = max(1, min(STEP, int((255 - 25) / len(group))))
            dst_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
            for j, contour in enumerate(group):
                tmp_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
                tmp_label = cv2.drawContours(tmp_label, [contour], -1, 255, -1)

                # 膨胀补上边界
                dilation_kernel = np.ones((3, 3), np.uint8)
                tmp_label = cv2.dilate(tmp_label, dilation_kernel, iterations=2)

                tmp_contours, _ = cv2.findContours(tmp_label,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                dst_label = cv2.drawContours(dst_label, tmp_contours, -1,
                                             color=255 - j * step, thickness=-1)

            transform = get_transform(image_size, dst_label.shape[0], dst_label.shape[1])
            dst_label = transform(image=dst_label)["image"]

            if i == 0:
                label_name = basename
            else:
                label_name = f"{basename[:-4]}({i}).png"

            dst_path = os.path.join(dst_label_dir, label_name)
            cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--image_size", type=int)
    args = parser.parse_args()

    preprocess(args.src_root, args.dst_root, args.image_size)
