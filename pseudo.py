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
import time
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

    def __init__(self, schedular_dir: str, focused_metric: str,
                 initial_sample_rate: float, sample_rate_delta: float,
                 metric_delta_threshold: float, current_epoch: int, step: int,
                 start_epoch: int, pseudo_weight: float,
                 pseudo_weight_gr: float = 0.0):
        self.schedular_dir = schedular_dir
        self.focused_metric = focused_metric
        self._sample_rates = {
            "initial": {"step": 0, "value": initial_sample_rate},
            "last": {"step": None, "value": None},
            "current": {"step": None, "value": None},
        }
        self.sample_rate_delta = sample_rate_delta
        self.metric_delta_threshold = metric_delta_threshold
        self.metric_data = {
            "last": {"step": None, "value": None},
            "current": {"step": None, "value": None}
        }
        self.current_epoch = current_epoch
        self.start_epoch = start_epoch
        self._step = step
        self._current_step = 0
        self._pseudo_weight = pseudo_weight
        self.pseudo_weight_gr = pseudo_weight_gr
        self._schedular_path = os.path.join(self.schedular_dir, "schedular.json")

        if os.path.exists(self._schedular_path):
            with open(self._schedular_path, "r") as f:
                self._schedular_data = json.load(f)
        else:
            self._schedular_data = {"schedular": [], "skip": []}

        # update using init args
        self._schedular_data["current_epoch"] = self.current_epoch
        self._schedular_data["start_epoch"] = self.start_epoch
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
        schedular_data["current_epoch"] = self.current_epoch
        schedular_data["step"] = self._step
        with open(self._schedular_path, "w") as f:
            json.dump(schedular_data, f, indent=2)

    @property
    def pseudo_weight(self):
        return max(0, min(self._pseudo_weight + self.pseudo_weight_gr * (self.current_epoch - self.start_epoch + 1), 1))

    @property
    def sample_rate(self, debug=False):
        if not self.is_active():
            return 0.0

        if self._current_step != self._sample_rates["current"]["step"]:
            last_val = self.metric_data["last"].get("value") or 0.0
            current_val = self.metric_data["current"].get("value") or 0.0
            delta = current_val - last_val

            if delta > self.metric_delta_threshold:
                delta_sample_rate = self.sample_rate_delta
            elif delta <= self.metric_delta_threshold:
                delta_sample_rate = -1 * self.sample_rate_delta
            else:
                delta_sample_rate = 0

            initial_rate = self._sample_rates["initial"]["value"]
            last_sample_rate = self._sample_rates["last"].get("value") or initial_rate
            sample_rate = last_sample_rate + delta_sample_rate
            if debug:
                print(f"\n====>>>>last_val: {last_val}, current_val: {current_val}, delta: {delta}, delta_sample_rate: {delta_sample_rate}, sample_rate: {sample_rate}")


            if debug:
                print(f"\n====>>>>befort self._sample_rates: {self._sample_rates}")
            self._sample_rates["last"]["step"] = self._sample_rates["current"]["step"]
            self._sample_rates["last"]["value"] = self._sample_rates["current"]["value"]
            self._sample_rates["current"] = {
                "step": self._current_step,
                "value": min(1.0, max(initial_rate, sample_rate))
            }

            if debug:
                print(f"\n====>>>>after self._sample_rates: {self._sample_rates}")


        return self._sample_rates["current"]["value"]

    def step(self, update_epoch: bool = False):
        if update_epoch:
            self.current_epoch += 1
        else:
            self._current_step += 1
        self._update()
        return self

    def update_metrics(self, metrics_data: dict = None):
        print("self.metric_data before updated: ", self.metric_data)
        self.metric_data["last"] = self.metric_data["current"]
        self.metric_data["current"] = {
            "step": self._current_step,
            "value": metrics_data[self.focused_metric]
        }
        print("self.metric_data after updated: ", self.metric_data)

    def is_active(self):
        schedular_data = self._read()
        if self.current_epoch in schedular_data["schedular"]:
            active = True
        elif self.current_epoch < self.start_epoch:
            active = False
        elif self.current_epoch not in schedular_data["skip"]:
            if self.start_epoch == 0:
                if self.current_epoch % self._step == 0:
                    active = True
                else:
                    active = False
            else:
                if self.current_epoch != 0 and self.current_epoch % self._step == 0:
                    active = True
                else:
                    active = False
        else:
            active = False
        return active


def write_info(info_json: str, info: dict, lock):
    with lock:
        if os.path.exists(info_json):
            with open(info_json) as f:
                json_data = json.load(f)
        else:
            json_data = {}

        for k, v in info.items():
            if k not in json_data:
                json_data[k] = v
            else:
                json_data[k] += v

        if json_data:
            with open(info_json, "w") as f:
                json.dump(json_data, f)

        return json_data


def read_info(info_json: str, lock):
    with lock:
        if os.path.exists(info_json):
            with open(info_json) as f:
                return json.load(f)
        else:
            return {}


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


def progress(info_json: str, total: int, lock):
    with tqdm(total=total, desc="generate pseudo masks") as pbar:
        while True:
            time.sleep(10)
            info_data = read_info(info_json, lock)
            if not info_data:
                continue
            info_total = info_data["total"]
            pbar.update(info_total - pbar.n)
            if info_total == total:   # >?
                break


@torch.no_grad()
def generate_pseudo(args, model, img_paths: List[str], pseudo_root: str,
                    info_path: str, lock, n_save: int = 100,
                    save_png_mask: bool = False):
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

    info = {"total": 0, "instances": 0, "empty": 0}
    for i, path in enumerate(img_paths):

        info["total"] += 1

        if i != 0 and i % n_save == 0:
            task_info = write_info(info_path, info, lock)
            info = {"total": 0, "instances": 0, "empty": 0}

        image = cv2.imread(path)
        if image is None:
            print(f"Could not load '{path}' as an image, skipping...")
            continue

        # data
        transform = get_transform(args.image_size, image.shape[0], image.shape[1])
        dst_img = transform(image=image)["image"]
        dst_path = os.path.join(pseudo_data_dir, os.path.basename(path))

        # mask
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = mask_predictor.predict(image)

        vals = [_ for _ in np.unique(mask) if _ != 0]
        # print("path = ", path, len(vals))
        if len(vals) == 0:
            info["empty"] += 1
            continue

        info["instances"] += len(vals)
        cv2.imwrite(dst_path, dst_img)

        basename = os.path.basename(path)[:-4]
        mask_path = os.path.join(pseudo_label_dir, f"{basename}.npy")

        # npy
        dst_arr = transform(image=mask)["image"]
        np.save(mask_path, dst_arr)

        # png
        if save_png_mask:
            img_path = os.path.join(pseudo_label_dir, f"{basename}.png")
            vals_uint8 = vals[:255]
            dst_label_uint8 = np.zeros(shape=dst_arr.shape, dtype=np.uint8)
            if len(vals_uint8) > 0:
                random.shuffle(vals_uint8)
                step = calc_step(len(vals_uint8))
                for j, val in enumerate(vals_uint8):
                    dst_label_uint8[dst_arr == val] = 255 - j * step

            cv2.imwrite(img_path, dst_label_uint8)

    if info["total"] > 0:
        write_info(info_path, info, lock)


def read_pseudo_info(info_path: str) -> dict:

    with open(info_path) as f:
        info = json.load(f)

    info["miss_rate"] = info["empty"] / info["total"] if info["total"] else 0.0
    info["average_instances"] = info["instances"] / info["total"] if info["total"] else 0.0

    return info


@torch.no_grad()
def generate_pseudo_multiple(args, model, pseudo_root: str):

    if os.path.exists(pseudo_root):
        shutil.rmtree(pseudo_root)

    os.makedirs(pseudo_root)
    pseudo_data_dir = os.path.join(pseudo_root, "data")
    pseudo_label_dir = os.path.join(pseudo_root, "label")
    os.makedirs(pseudo_data_dir)
    os.makedirs(pseudo_label_dir)

    img_paths = glob.glob(os.path.join(args.unsupervised_dir, "*.png"))
    tasks = split_list(img_paths, args.unsupervised_num_processes)

    lock = mp.Lock()
    info_path = os.path.join(pseudo_root, "info.json")

    n_save = 100

    processes = []
    for task in tasks:
        p = mp.Process(
            target=generate_pseudo,
            args=(args, model, task, pseudo_root, info_path, lock, n_save, False)
        )
        p.start()
        processes.append(p)

    p = mp.Process(target=progress, args=(info_path, len(img_paths), lock))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    pseudo_info = read_pseudo_info(info_path)

    split_path = os.path.join(pseudo_root, "split.json")
    split_dataset(data_root=pseudo_root, ext="png", test_size=0.0,
                  split_path=split_path)

    return split_path, pseudo_info
