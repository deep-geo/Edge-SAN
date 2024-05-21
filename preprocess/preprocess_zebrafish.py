"""
Script for zebrafish dataset preprocessing.

Original dataset structure:
.
├── Mouse(NucMM-M)
│   ├── Image
│   │   ├── train
│   │   │   ├── img_000_200_200.h5
│   │   │   ├── img_000_604_576.h5
│   │   │   ├── img_508_200_576.h5
│   │   │   └── img_508_604_200.h5
│   │   └── val
│   │       ├── img_000_200_576.h5
│   │       ├── img_000_604_200.h5
│   │       ├── img_508_200_200.h5
│   │       └── img_508_604_576.h5
│   ├── Label
│   │   ├── train
│   │   │   ├── seg_000_200_200.h5
│   │   │   ├── seg_000_604_576.h5
│   │   │   ├── seg_508_200_576.h5
│   │   │   └── seg_508_604_200.h5
│   │   └── val
│   │       ├── seg_000_200_576.h5
│   │       ├── seg_000_604_200.h5
│   │       ├── seg_508_200_200.h5
│   │       └── seg_508_604_576.h5
│   ├── README.txt
│   └── image.h5
└── Zebrafish(NucMM-Z)
    ├── Image
    │   ├── train
    │   │   ├── img_0000_0576_0768.h5
    │   │   ├── img_0000_0704_0832.h5
    │   │   ├── img_0064_0384_0832.h5
    │   │   ...
    │   └── val
    │       ├── img_0000_0640_0832.h5
    │       ├── img_0000_0640_0896.h5
    │       ├── img_0000_0768_1024.h5
    │       ...
    ├── Label
    │   ├── train
    │   │   ├── seg_0000_0576_0768.h5
    │   │   ├── seg_0000_0704_0832.h5
    │   │   ├── seg_0064_0384_0832.h5
    │   │   ...
    │   └── val
    │       ├── seg_0000_0640_0832.h5
    │       ├── seg_0000_0640_0896.h5
    │       ├── seg_0000_0768_1024.h5
    │       ...
    ├── README.txt
    ├── image.tif
    └── mask.h5
"""
import os
import glob
import argparse
import numpy as np
import tqdm
import h5py

from preprocess import Preprocess


class PreprocessZebrafish(Preprocess):

    def process(self):

        print("\nProcess Mouse(NucMM-M)...")
        src_data_dir = os.path.join(self.src_root, "Mouse(NucMM-M)", "Image")
        src_label_dir = os.path.join(self.src_root, "Mouse(NucMM-M)", "Label")
        for split in ["train", "val"]:
            src_data_paths = glob.glob(os.path.join(src_data_dir, split, "*.h5"))
            for src_data_path in tqdm.tqdm(src_data_paths):
                data_basename = os.path.basename(src_data_path)
                label_basename = "seg_" + data_basename[4:]
                src_label_path = os.path.join(src_label_dir, split, label_basename)
                images = h5py.File(src_data_path, "r")
                labels = h5py.File(src_label_path, "r")
                images = np.array(images['main'])
                labels = np.array(labels['main'])

                for dim in range(3):
                    for i in range(images.shape[dim]):
                        basename = data_basename[:-3] + f"_slice{i:03d}" + f"_dim{dim}"

                        if dim == 0:
                            img = images[i, :, :]
                            label = labels[i, :, :]
                        elif dim == 1:
                            img = images[:, i, :]
                            label = labels[:, i, :]
                        else:
                            img = images[:, :, i]
                            label = labels[:, :, i]

                        self.save_data(ori_data=img,
                                       data_name=f"Mouse_{basename}")
                        self.save_label(ori_label=label,
                                        label_name=f"Mouse_{basename}")

        print("\nProcess Zebrafish(NucMM-Z)...")
        src_data_dir = os.path.join(self.src_root, "Zebrafish(NucMM-Z)", "Image")
        src_label_dir = os.path.join(self.src_root, "Zebrafish(NucMM-Z)", "Label")
        for split in ["train", "val"]:
            src_data_paths = glob.glob(
                os.path.join(src_data_dir, split, "*.h5"))
            for src_data_path in tqdm.tqdm(src_data_paths):
                data_basename = os.path.basename(src_data_path)
                label_basename = "seg_" + data_basename[4:]
                src_label_path = os.path.join(src_label_dir, split,
                                              label_basename)
                images = h5py.File(src_data_path, "r")
                labels = h5py.File(src_label_path, "r")
                images = np.array(images['main'])
                labels = np.array(labels['main'])

                for dim in range(3):
                    for i in range(images.shape[dim]):
                        basename = data_basename[:-3] + f"_slice{i:03d}" + f"_dim{dim}"

                        if dim == 0:
                            img = images[i, :, :]
                            label = labels[i, :, :]
                        elif dim == 1:
                            img = images[:, i, :]
                            label = labels[:, i, :]
                        else:
                            img = images[:, :, i]
                            label = labels[:, :, i]

                        self.save_data(ori_data=img,
                                       data_name=f"Zebrafish_{basename}")
                        self.save_label(ori_label=label,
                                        label_name=f"Zebrafish_{basename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessZebrafish(args.src_root, args.dst_root,
                        args.dst_size, args.dst_prefix).process()
