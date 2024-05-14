"""
Script for DynamicNuclearNet dataset preprocessing.

Original dataset structure:
.
├── LICENSE
├── README.md
├── test.npz
├── train.npz
└── val.npz

"""
import os
import argparse
import cv2
import numpy as np
import tqdm
import random

from preprocess import Preprocess


class PreprocessDynamicNuclearNet(Preprocess):

    def process(self):

        counter = {}

        for split in ["train", "test", "val"]:
            print(f"\nProcess {split} data...")

            src_data_path = os.path.join(self.src_root, f"{split}.npz")
            src_data = np.load(src_data_path, allow_pickle=True)
            X = src_data["X"]
            y = src_data["y"]
            filenames = src_data["meta"][1:, 0]

            for i in tqdm.tqdm(range(len(filenames))):
                img_16bit = X[i, :, :, 0]
                img_normalized = ((img_16bit - img_16bit.min()) /
                                  (img_16bit.max() - img_16bit.min()))
                img = (img_normalized * 255).astype(np.uint8)

                label = y[i, :, :, 0]

                filename = filenames[i]
                filename = os.path.basename(filename)[:-4]
                if filename not in counter:
                    counter[filename] = 1
                else:
                    counter[filename] += 1

                dst_filename = self.dst_prefix + f"{filename}_{counter[filename]}.png"

                # data
                dst_img = self.transform(img)
                dst_data_path = os.path.join(self.dst_data_dir, dst_filename)
                cv2.imwrite(dst_data_path, dst_img)

                # label
                vals = [_ for _ in np.unique(label) if _ != 0]
                random.shuffle(vals)
                val_groups = [vals[i:i + 255] for i in range(0, len(vals), 255)]
                for j, group in enumerate(val_groups):
                    step = self.calc_step(len(group))
                    dst_label = np.zeros(
                        shape=(label.shape[0], label.shape[1]),
                        dtype=np.uint8
                    )
                    for k, val in enumerate(group):
                        tmp_label = label.copy().astype(np.uint8)
                        tmp_label[label != val] = 0
                        tmp_label[label == val] = 255
                        tmp_contours, _ = cv2.findContours(tmp_label,
                                                           cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                        dst_label = cv2.drawContours(dst_label, tmp_contours,
                                                     -1,
                                                     color=255 - k * step,
                                                     thickness=-1)

                    dst_label = self.transform(dst_label)

                    if j == 0:
                        label_name = dst_filename
                    else:
                        label_name = f"{dst_filename[:-4]}({j}).png"

                    dst_path = os.path.join(self.dst_label_dir, label_name)
                    cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessDynamicNuclearNet(args.src_root, args.dst_root,
                                args.dst_size, args.dst_prefix).process()
