import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import random
import cv2
import numpy as np

from abc import abstractmethod
from utils import calc_step, train_transforms, get_boxes_from_mask


class Preprocess0:

    def __init__(self, src_root: str, dst_root: str, dst_size: int,
                 dst_prefix: str = None):
        self.src_root = src_root

        self.dst_root = dst_root
        self.dst_data_dir = os.path.join(self.dst_root, "data")
        self.dst_label_dir = os.path.join(self.dst_root, "label")
        os.makedirs(self.dst_data_dir, exist_ok=True)
        os.makedirs(self.dst_label_dir, exist_ok=True)

        self.dst_size = int(dst_size)

        self.dst_prefix = "" if dst_prefix is None else dst_prefix.strip()
        self.dst_prefix = self.dst_prefix + "_" if self.dst_prefix \
            else self.dst_prefix

    def save_data(self, data: np.ndarray, data_name: str):
        if len(data.shape) == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        dst_img_path = os.path.join(
            self.dst_data_dir,
            self.dst_prefix + data_name + ".png"
        )
        cv2.imwrite(dst_img_path, data)

    def save_label(self, label: np.ndarray, label_name: str):
        label_uint16 = label.astype(np.uint16)

        # npy
        dst_arr_path = os.path.join(
            self.dst_label_dir,
            self.dst_prefix + label_name + ".npy"
        )
        np.save(dst_arr_path, label_uint16)

        # png
        vals_uint16 = [_ for _ in np.unique(label_uint16) if _ != 0][:255]
        random.shuffle(vals_uint16)
        dst_label_uint8 = np.zeros(shape=label_uint16.shape, dtype=np.uint8)
        if len(vals_uint16) > 0:
            step = calc_step(len(vals_uint16))
            for j, val in enumerate(vals_uint16):
                dst_label_uint8[label_uint16 == val] = 255 - j * step
        dst_img_path = os.path.join(
            self.dst_label_dir,
            self.dst_prefix + label_name + ".png"
        )
        cv2.imwrite(dst_img_path, dst_label_uint8)

    @abstractmethod
    def process(self):
        raise NotImplementedError


# For SAM MED2D Baseline
class Preprocess:

    def __init__(self, src_root: str, dst_root: str, dst_size: int,
                 dst_prefix: str = None):
        self.src_root = src_root

        self.dst_root = dst_root
        self.dst_data_dir = os.path.join(self.dst_root, "data")
        self.dst_label_dir = os.path.join(self.dst_root, "label")
        os.makedirs(self.dst_data_dir, exist_ok=True)
        os.makedirs(self.dst_label_dir, exist_ok=True)

        self.dst_size = int(dst_size)

        self.dst_prefix = "" if dst_prefix is None else dst_prefix.strip()
        self.dst_prefix = self.dst_prefix + "_" if self.dst_prefix \
            else self.dst_prefix

    def save_data(self, data: np.ndarray, data_name: str):
        if len(data.shape) == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        dst_img_path = os.path.join(
            self.dst_data_dir,
            self.dst_prefix + data_name + ".png"
        )
        cv2.imwrite(dst_img_path, data)

    def save_label(self, label: np.ndarray, label_name: str):
        vals = [_ for _ in np.unique(label) if _ != 0][:255]
        for i, val in enumerate(vals):
            mask = np.zeros(shape=label.shape, dtype=np.uint8)
            mask[label == val] = 255

            transforms = train_transforms(self.dst_size, mask.shape[0],
                                          mask.shape[1])
            augments = transforms(image=mask)
            mask = augments['image']
            dst_img_path = os.path.join(
                self.dst_label_dir,
                f"{label_name}_{i:05d}.png"
            )
            try:
                get_boxes_from_mask(mask, max_pixel=0)
            except:
                print(f"fail: {dst_img_path}")
                continue

            cv2.imwrite(dst_img_path, mask)

    @abstractmethod
    def process(self):
        raise NotImplementedError