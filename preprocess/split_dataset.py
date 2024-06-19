import os
import json
import random
import glob
import argparse

from typing import List


def split_dataset(data_root: str, ext: str, test_size: float, seed: int = None,
                  split_path: str = None):

    if split_path is None:
        split_path = os.path.join(data_root, "split.json")

    if ext.startswith("."):
        ext = ext[1:]

    data_dir = os.path.join(data_root, "data")
    label_dir = os.path.join(data_root, "label")

    data_paths: List[str] = []
    label_paths: List[str] = []
    for data_path in glob.glob(os.path.join(data_dir, f"*.{ext}")):
        basename = os.path.basename(data_path)
        label_path = os.path.join(label_dir, basename[:-4] + ".npy")
        if os.path.exists(label_path):
            data_paths.append(os.path.relpath(data_path, data_root))
            label_paths.append(os.path.relpath(label_path, data_root))

    if seed is not None:
        random.seed(seed)

    indices = list(range(len(data_paths)))
    random.shuffle(indices)

    total = len(data_paths)
    num_test = int(total * test_size)
    test_data_paths = [data_paths[i] for i in indices[:num_test]]
    test_label_paths = [label_paths[i] for i in indices[:num_test]]

    train_data_paths = [data_paths[i] for i in indices[num_test:]]
    train_label_paths = [label_paths[i] for i in indices[num_test:]]

    split_json = {
        "seed": seed,
        "test_size": test_size,
        "quantity": {"train": total - num_test, "test": num_test, "total": total},
        "train": [(dp, lp) for dp, lp in zip(train_data_paths, train_label_paths)],
        "test": [(dp, lp) for dp, lp in zip(test_data_paths, test_label_paths)]
    }

    with open(split_path, "w") as f:
        json.dump(split_json, f, indent=2)

    # print(f"Save split json to: {split_path}")

    return split_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--ext", type=str, default="png")
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.data_root, args.ext, args.test_size, args.seed)
