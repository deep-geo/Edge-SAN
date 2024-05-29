import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import random
import json
import cv2
import numpy as np

from abc import abstractmethod
from typing import Dict
from utils import calc_step, get_transform


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
        self.dst_prefix = self.dst_prefix + "_" if self.dst_prefix else self.dst_prefix

        self.count = 0
        self.nuclei = 0

    def _get_transform(self, ori_h: int, ori_w: int):
        return get_transform(self.dst_size, ori_h, ori_w)

    def transform(self, img: np.ndarray):
        # img shape: h * w * c
        t = self._get_transform(img.shape[0], img.shape[1])
        return t(image=img)["image"]

    def calc_step(self, num_colors: int):
        return calc_step(num_colors)

    def save_data(self, ori_data: np.ndarray, data_name: str):
        dst_img = self.transform(ori_data)
        dst_img_path = os.path.join(
            self.dst_data_dir,
            self.dst_prefix + data_name + ".png"
        )
        cv2.imwrite(dst_img_path, dst_img)

    def save_label(self, ori_label: np.ndarray, label_name: str):
        self.count += 1
        self.nuclei += len(np.unique(ori_label)) - 1
        label = ori_label.astype(np.uint16)
        dst_label_uint16 = self.transform(label)

        # npy
        dst_arr_path = os.path.join(self.dst_label_dir,
                                    self.dst_prefix + label_name + ".npy")
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
                                    self.dst_prefix + label_name + ".png")
        cv2.imwrite(dst_img_path, dst_label_uint8)

    def save_info(self):
        if self.dst_prefix:
            info_key = self.dst_prefix[:-1]
        else:
            info_key = os.path.basename(os.path.abspath(self.src_root))

        info_path = os.path.join(self.dst_root, "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
                info[info_key] = {"count": self.count, "nuclei": self.nuclei}
        else:
            info = {info_key: {"count": self.count, "nuclei": self.nuclei}}

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    @abstractmethod
    def process(self):
        raise NotImplementedError
