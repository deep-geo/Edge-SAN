import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

from abc import abstractmethod
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

    def _get_transform(self, ori_h: int, ori_w: int):
        return get_transform(self.dst_size, ori_h, ori_w)

    def transform(self, img: np.ndarray):
        # img shape: h * w * c
        t = self._get_transform(img.shape[0], img.shape[1])
        return t(image=img)["image"]

    def calc_step(self, num_colors: int):
        return calc_step(num_colors)

    @abstractmethod
    def process(self):
        raise NotImplementedError
