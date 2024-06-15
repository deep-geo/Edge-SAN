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
import shutil
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

    def __init__(self, schedular_dir: str, sample_rates: List[float],
                 current_step: int, step: int, start_step: int,
                 pseudo_weight_gr: float):
        self.schedular_dir = schedular_dir
        self.sample_rates = sample_rates
        self.current_step = current_step
        self.start_step = start_step
        self._step = step
        self.pseudo_weight_gr = pseudo_weight_gr
        self._schedular_path = os.path.join(self.schedular_dir, "schedular.json")

        if os.path.exists(self._schedular_path):
            with open(self._schedular_path, "r") as f:
                self._schedular_data = json.load(f)
        else:
            self._schedular_data = {"schedular": [], "skip": []}

        # update using init args
        self._schedular_data["current_epoch"] = self.current_step
        self._schedular_data["start_epoch"] = self.start_step
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
        schedular_data["current_epoch"] = self.current_step
        schedular_data["step"] = self._step
        with open(self._schedular_path, "w") as f:
            json.dump(schedular_data, f, indent=2)

    @property
    def pseudo_weight(self):
        return max(0, min(self.pseudo_weight_gr * (self.current_step - self.start_step + 1), 1))

    @property
    def sample_rate(self):
        idx = self.current_step - self.start_step
        if idx < 0:
            rate = 0
        elif idx < len(self.sample_rates):
            rate = self.sample_rates[idx]
        else:
            rate = self.sample_rates[-1]
        return rate

    def step(self):
        self.current_step += 1
        self._update()
        return self

    def is_active(self):
        schedular_data = self._read()
        if self.current_step in schedular_data["schedular"]:
            active = True
        elif self.current_step < self.start_step:
            active = False
        elif self.current_step not in schedular_data["skip"]:
            if self.start_step == 0:
                if self.current_step % self._step == 0:
                    active = True
                else:
                    active = False
            else:
                if self.current_step != 0 and self.current_step % self._step == 0:
                    active = True
                else:
                    active = False
        else:
            active = False
        return active


@torch.no_grad()
def generate_pseudo(args, model, pseudo_root: str, img_paths: List[str] = None,
                    task_id: int = None, save_png_mask: bool = False,
                    split_path: str = None):
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
    if os.path.exists(pseudo_data_dir):
        shutil.rmtree(pseudo_data_dir)
    if os.path.exists(pseudo_label_dir):
        shutil.rmtree(pseudo_label_dir)
    os.makedirs(pseudo_data_dir)
    os.makedirs(pseudo_label_dir)

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

        # npy
        dst_arr = transform(image=mask)["image"]
        np.save(mask_path, dst_arr)

        # png
        if save_png_mask:
            img_path = os.path.join(pseudo_label_dir, f"{basename}.png")
            vals_uint16 = [_ for _ in np.unique(dst_arr) if _ != 0][:255]
            dst_label_uint8 = np.zeros(shape=dst_arr.shape, dtype=np.uint8)
            if len(vals_uint16) > 0:
                random.shuffle(vals_uint16)
                step = calc_step(len(vals_uint16))
                for j, val in enumerate(vals_uint16):
                    dst_label_uint8[dst_arr == val] = 255 - j * step

            cv2.imwrite(img_path, dst_label_uint8)

    split_dataset(data_root=pseudo_root, ext="png", test_size=0.0, split_path=split_path)


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
def generate_pseudo_multiple(args, model, pseudo_root: str, sample_rate: float = 1.0):

    assert 0 < sample_rate <= 1.0, "wrong sample_rate!"

    img_paths = glob.glob(os.path.join(args.unsupervised_dir, "*.png"))
    print(f"\noriginal pseudo number: {len(img_paths)}")
    img_paths = random.choices(img_paths, k=int(len(img_paths) * sample_rate))
    print(f"\npseudo sample_rate = {sample_rate}, actual pseudo sample number: {len(img_paths)}")
    tasks = split_list(img_paths, args.unsupervised_num_processes)

    processes = []
    split_paths = []
    for i, task in enumerate(tasks):
        pseudo_dir = os.path.join(pseudo_root, f"split_{i}")
        split_path = os.path.join(pseudo_dir, "split.json")
        split_paths.append(split_path)
        p = mp.Process(target=generate_pseudo,
                       args=(args, model, pseudo_dir, task, i, False, split_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return split_paths
