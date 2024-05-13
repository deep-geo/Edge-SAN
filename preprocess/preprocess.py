import os
import cv2
import numpy as np
import albumentations as A
import glob

from abc import abstractmethod


STEP = 15


class Preprocess:

    def __init__(self, src_root: str, dst_root: str, dst_size: int,
                 dst_prefix: str = None):
        self.src_root = src_root

        self.dst_root = dst_root
        if os.path.exists(dst_root):
            assert not glob.glob(os.path.join(dst_root, "*")), \
                f"The dst_root is supposed to be empty: {dst_root}"

        self.dst_data_dir = os.path.join(self.dst_root, "data")
        self.dst_label_dir = os.path.join(self.dst_root, "label")
        os.makedirs(self.dst_data_dir)
        os.makedirs(self.dst_label_dir)

        self.dst_size = int(dst_size)
        self.dst_prefix = "" if dst_prefix is None else dst_prefix.strip()
        self.dst_prefix = self.dst_prefix + "_" if self.dst_prefix else self.dst_prefix

    def _get_transform(self, ori_h: int, ori_w: int):
        if ori_h < self.dst_size and ori_w < self.dst_size:
            t = A.PadIfNeeded(
                min_height=self.dst_size, min_width=self.dst_size,
                position="center", border_mode=cv2.BORDER_CONSTANT,
                value=0
            )
        else:
            t = A.Resize(
                self.dst_size, self.dst_size,
                interpolation=cv2.INTER_NEAREST
            )
        return t

    def transform(self, img: np.ndarray):
        # img shape: h * w * c
        t = self._get_transform(img.shape[0], img.shape[1])
        return t(image=img)["image"]

    def calc_step(self, num_colors: int):
        return max(1, min(STEP, int((255 - 25) / num_colors)))

    @abstractmethod
    def process(self):
        raise NotImplementedError
