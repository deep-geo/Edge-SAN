"""
Script for fluorescence dataset preprocessing.

Dataset: https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation

Original dataset structure:
.
├── Grade.csv
├── testA_1.bmp
├── testA_10.bmp
├── testA_10_anno.bmp
├── testA_11.bmp
├── testA_11_anno.bmp
├── testA_12.bmp
├── testA_12_anno.bmp
...

"""
import os
import glob
import argparse
import cv2
import numpy as np
import tqdm
import random

from preprocess import Preprocess


class PreprocessGlandSeg(Preprocess):

    def process(self):
        src_label_paths = glob.glob(os.path.join(self.src_root, "*_anno.bmp"))
        src_data_paths = []
        for path in src_label_paths:
            data_name = os.path.basename(path).replace("_anno", "")
            src_data_paths.append(os.path.join(self.src_root, data_name))

        # data
        print("\nProcess data...")
        for i, src_data_path in tqdm.tqdm(enumerate(src_data_paths),
                                          total=len(src_data_paths)):
            img = cv2.imread(src_data_path)
            dst_img = self.transform(img)
            dst_data_path = os.path.join(
                self.dst_data_dir,
                self.dst_prefix + f"image_{i + 1:04d}.png"
            )
            cv2.imwrite(dst_data_path, dst_img)

        # label
        print("\nProcess label...")
        for i, path in tqdm.tqdm(enumerate(src_label_paths),
                                 total=len(src_label_paths)):
            label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = label.astype(np.uint16)    # keep the same data type with other datasets

            dst_label_uint16 = self.transform(label)

            basename = f"image_{i + 1:04d}"

            # npy
            dst_arr_path = os.path.join(self.dst_label_dir,
                                        self.dst_prefix + f"{basename}.npy")
            np.save(dst_arr_path, dst_label_uint16)

            # png
            vals_uint16 = [_ for _ in np.unique(dst_label_uint16) if _ != 0][
                          :255]
            dst_label_uint8 = np.zeros(shape=dst_label_uint16.shape,
                                       dtype=np.uint8)
            if len(vals_uint16) > 0:
                random.shuffle(vals_uint16)
                step = self.calc_step(len(vals_uint16))
                for j, val in enumerate(vals_uint16):
                    dst_label_uint8[dst_label_uint16 == val] = 255 - j * step

            dst_img_path = os.path.join(self.dst_label_dir,
                                        self.dst_prefix + f"{basename}.png")
            cv2.imwrite(dst_img_path, dst_label_uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessGlandSeg(args.src_root, args.dst_root,
                       args.dst_size, args.dst_prefix).process()

