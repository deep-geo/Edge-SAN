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
            src_label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            vals = [_ for _ in np.unique(src_label) if _ != 0]
            random.shuffle(vals)
            val_groups = [vals[i:i + 255] for i in range(0, len(vals), 255)]
            for j, group in enumerate(val_groups):
                step = self.calc_step(len(group))
                dst_label = np.zeros(
                    shape=(src_label.shape[0], src_label.shape[1]),
                    dtype=np.uint8
                )
                for k, val in enumerate(group):
                    tmp_label = src_label.copy().astype(np.uint8)
                    tmp_label[src_label != val] = 0
                    tmp_label[src_label == val] = 255
                    tmp_contours, _ = cv2.findContours(tmp_label,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    dst_label = cv2.drawContours(dst_label, tmp_contours, -1,
                                                 color=255 - k * step,
                                                 thickness=-1)

                dst_label = self.transform(dst_label)

                if j == 0:
                    label_name = self.dst_prefix + f"image_{i + 1:04d}.png"
                else:
                    label_name = self.dst_prefix + f"image_{i + 1:04d}({j}).png"

                dst_path = os.path.join(self.dst_label_dir, label_name)
                cv2.imwrite(dst_path, dst_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessGlandSeg(args.src_root, args.dst_root,
                       args.dst_size, args.dst_prefix).process()
