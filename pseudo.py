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
import math
import os
import json
import random
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from typing import List
from tqdm import tqdm
from preprocess.split_dataset import split_dataset
from utils import get_transform, calc_step, MaskPredictor


class PseudoSchedular:

    def __init__(self, schedular_dir: str, current_epoch: int, step: int,
                 start_epoch: int, pseudo_weight_gr: float):
        self._schedular_dir = schedular_dir
        self._current_epoch = current_epoch
        self._start_epoch = start_epoch
        self._step = step
        self._pseudo_weight_gr = pseudo_weight_gr
        self._schedular_path = os.path.join(self._schedular_dir, "schedular.json")

        if os.path.exists(self._schedular_path):
            with open(self._schedular_path, "r") as f:
                self._schedular_data = json.load(f)
        else:
            self._schedular_data = {"schedular": [], "skip": []}

        # update using init args
        self._schedular_data["current_epoch"] = self._current_epoch
        self._schedular_data["start_epoch"] = self._start_epoch
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
        schedular_data["step"] = self._step
        with open(self._schedular_path, "w") as f:
            json.dump(schedular_data, f, indent=2)

    @property
    def pseudo_weight(self):
        return min(self._pseudo_weight_gr * (self._current_epoch - self._start_epoch + 1), 1)

    def step(self):
        self._current_epoch += 1
        self._update()
        return self

    def is_active(self):
        schedular_data = self._read()
        if self._current_epoch in schedular_data["schedular"]:
            active = True
        elif self._current_epoch < self._start_epoch:
            active = False
        elif self._current_epoch not in schedular_data["skip"]:
            if self._start_epoch == 0:
                if self._current_epoch % self._step == 0:
                    active = True
                else:
                    active = False
            else:
                if self._current_epoch != 0 and self._current_epoch % self._step == 0:
                    active = True
                else:
                    active = False
        else:
            active = False
        return active


@torch.no_grad()
def generate_pseudo(args, model, pseudo_root: str, img_paths: List[str] = None,
                    task_id: int = None):
    model.eval()
    mask_predictor = MaskPredictor(
        model=model,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=10,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch
    )
    pseudo_data_dir = os.path.join(pseudo_root, "data")
    pseudo_label_dir = os.path.join(pseudo_root, "label")
    os.makedirs(pseudo_data_dir, exist_ok=True)
    os.makedirs(pseudo_label_dir, exist_ok=True)

    if not img_paths:
        img_paths = glob.glob(os.path.join(args.unsupervised_dir, "*.png"))

    desc = "Generating pseudo masks"
    if task_id is not None:
        desc = f"{desc}, task_id: {task_id}"

    for path in tqdm(img_paths, desc=desc):
        image = cv2.imread(path)
        if image is None:
            print(f"Could not load '{path}' as an image, skipping...")
            continue

        # data
        transform = get_transform(args.image_size, image.shape[0], image.shape[1])
        dst_img = transform(image=image)["image"]
        dst_path = os.path.join(pseudo_data_dir, os.path.basename(path))
        cv2.imwrite(dst_path, dst_img)

        # mask
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = mask_predictor.predict(image)

        basename = os.path.basename(path)[:-4]
        mask_path = os.path.join(pseudo_label_dir, f"{basename}.npy")
        img_path = os.path.join(pseudo_label_dir, f"{basename}.png")

        # npy
        dst_arr = transform(image=mask)["image"]
        np.save(mask_path, dst_arr)

        # png
        vals_uint16 = [_ for _ in np.unique(dst_arr) if _ != 0][:255]
        dst_label_uint8 = np.zeros(shape=dst_arr.shape, dtype=np.uint8)
        if len(vals_uint16) > 0:
            random.shuffle(vals_uint16)
            step = calc_step(len(vals_uint16))
            for j, val in enumerate(vals_uint16):
                dst_label_uint8[dst_arr == val] = 255 - j * step

        cv2.imwrite(img_path, dst_label_uint8)

    split_dataset(data_root=pseudo_root, ext="png", test_size=0.0)


def split_list(lst, num_parts):

    avg_len = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0

    for i in range(num_parts):
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result


@torch.no_grad()
def generate_pseudo_multiple(args, model, pseudo_root: str):
    img_paths = glob.glob(os.path.join(args.unsupervised_dir, "*.png"))
    tasks = split_list(img_paths, args.unsupervised_num_processes)
    # print("task lengths: ", [len(_) for _ in tasks])

    processes = []
    split_paths = []
    for i, task in enumerate(tasks):
        pseudo_dir = os.path.join(pseudo_root, f"split_{i}")
        split_paths.append(os.path.join(pseudo_dir, "split.json"))
        p = mp.Process(target=generate_pseudo, args=(args, model, pseudo_dir, task, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return split_paths
