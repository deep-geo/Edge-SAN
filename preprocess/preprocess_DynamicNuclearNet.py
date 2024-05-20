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
                label = y[i, :, :, 0].astype(np.uint16)
                dst_label_uint16 = self.transform(label)

                # npy
                dst_arr_path = os.path.join(self.dst_label_dir,
                                            f"{dst_filename[:-4]}.npy")
                np.save(dst_arr_path, dst_label_uint16)

                # png
                vals_uint16 = [_ for _ in np.unique(dst_label_uint16) if _ != 0][:255]
                dst_label_uint8 = np.zeros(shape=dst_label_uint16.shape,
                                           dtype=np.uint8)
                if len(vals_uint16) > 0:
                    random.shuffle(vals_uint16)
                    step = self.calc_step(len(vals_uint16))
                    for j, val in enumerate(vals_uint16):
                        dst_label_uint8[dst_label_uint16 == val] = 255 - j * step

                dst_img_path = os.path.join(self.dst_label_dir, dst_filename)
                cv2.imwrite(dst_img_path, dst_label_uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessDynamicNuclearNet(args.src_root, args.dst_root,
                                args.dst_size, args.dst_prefix).process()
