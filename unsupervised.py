"""Schedular to control training status

Schedular json file:
{
    "current_epoch": xxx,
    "end_epoch": xxx,
    "step": xxx,
    "schedular": [xxx, xxx, ...],
    "skip": [xxx, xxx, ...]
}

"""
import glob
import os
import json
import random
import cv2
import numpy as np
import torch

from tqdm import tqdm
from preprocess.split_dataset import split_dataset
from segment_anything import SamAutomaticMaskGenerator
from utils import get_transform, calc_step


class Schedular:

    def __init__(self, schedular_dir: str, current_epoch: int, step: int,
                 start_epoch: int = 0, end_epoch: int = None):
        self._schedular_dir = schedular_dir
        self._current_epoch = current_epoch
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._step = step
        self._schedular_path = os.path.join(self._schedular_dir, "schedular.json")

        if os.path.exists(self._schedular_path):
            with open(self._schedular_path, "r") as f:
                self._schedular_data = json.load(f)
        else:
            self._schedular_data = {"schedular": [], "skip": []}

        # update using init args
        self._schedular_data["current_epoch"] = self._current_epoch
        self._schedular_data["start_epoch"] = self._start_epoch
        self._schedular_data["end_epoch"] = self._end_epoch
        self._schedular_data["step"] = self._step

    def _read(self):
        if not os.path.exists(self._schedular_path):
            with open(self._schedular_path, "w") as f:
                json.dump(self._schedular_data, f, indent=2)
            return self._schedular_data
        else:
            with open(self._schedular_path, "r") as f:
                return json.load(f)

    def _update(self):
        schedular_data = self._read()
        schedular_data["current_epoch"] = self._current_epoch
        schedular_data["end_epoch"] = self._end_epoch
        schedular_data["step"] = self._step
        with open(self._schedular_path, "w") as f:
            json.dump(schedular_data, f, indent=2)

    def step(self):
        self._current_epoch += 1
        self._update()
        return self

    def is_active(self):
        schedular_data = self._read()
        if self._current_epoch in schedular_data["schedular"]:
            active = True
        elif self._end_epoch is not None and self._current_epoch > self._end_epoch:
            active = False
        elif (self._current_epoch != 0 and self._current_epoch % self._step == 0) \
                and self._current_epoch not in schedular_data["skip"]:
            active = True
        else:
            active = False
        return active


@torch.no_grad()
def generate_unsupervised(args, model, dst_unsupervised_root: str):
    model.eval()
    generator = SamAutomaticMaskGenerator(
        model=model,
        pred_iou_thresh=args.unsupervised_pred_iou_thresh,
        stability_score_thresh=args.unsupervised_stability_score_thresh,
        stability_score_offset=1.0,
        box_nms_thresh=0.7,
        min_mask_region_area=10,
        points_per_batch=256
    )
    dst_data_dir = os.path.join(dst_unsupervised_root, "data")
    dst_label_dir = os.path.join(dst_unsupervised_root, "label")
    os.makedirs(dst_data_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(args.unsupervised_dir, "*.png"))

    for path in tqdm(img_paths, desc="Generating unsupervised mask"):
        image = cv2.imread(path)
        if image is None:
            print(f"Could not load '{path}' as an image, skipping...")
            continue

        # data
        transform = get_transform(args.image_size, image.shape[0], image.shape[1])
        dst_img = transform(image=image)["image"]
        dst_path = os.path.join(dst_data_dir, os.path.basename(path))
        cv2.imwrite(dst_path, dst_img)

        # mask
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image)
        arr = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint16)
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            arr[mask] = i + 1

        basename = os.path.basename(path)[:-4]
        arr_path = os.path.join(dst_label_dir, f"{basename}.npy")
        img_path = os.path.join(dst_label_dir, f"{basename}.png")

        # npy
        dst_arr = transform(image=arr)["image"]
        np.save(arr_path, dst_arr)

        # png
        vals_uint16 = [_ for _ in np.unique(dst_arr) if _ != 0][:255]
        dst_label_uint8 = np.zeros(shape=dst_arr.shape, dtype=np.uint8)
        if len(vals_uint16) > 0:
            random.shuffle(vals_uint16)
            step = calc_step(len(vals_uint16))
            for j, val in enumerate(vals_uint16):
                dst_label_uint8[dst_arr == val] = 255 - j * step

        cv2.imwrite(img_path, dst_label_uint8)

    split_dataset(data_root=dst_unsupervised_root, ext="png", test_size=0.0)
