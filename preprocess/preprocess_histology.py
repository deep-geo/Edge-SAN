"""
Script for histology dataset preprocessing.

Original dataset structure:
.
├── data
│   ├── 0TCGA-21-5784-01Z-00-DX1.png
│   ├── 0TCGA-21-5786-01Z-00-DX1.png
│   ├── 0TCGA-2Z-A9J9-01A-01-TS1.png
│   ...
└── label
    ├── 0TCGA-21-5784-01Z-00-DX1.png
    ├── 0TCGA-21-5786-01Z-00-DX1.png
    ├── 0TCGA-2Z-A9J9-01A-01-TS1.png
    ...
"""
import os
import glob
import argparse
import cv2
import tqdm
import numpy as np

from preprocess import Preprocess


class PreprocessHistology(Preprocess):

    def process(self):
        # data
        src_data_dir = os.path.join(self.src_root, "data")
        src_data_paths = glob.glob(os.path.join(src_data_dir, "*.png"))
        print("\nProcess data...")
        for path in tqdm.tqdm(src_data_paths):
            img = cv2.imread(path)
            self.save_data(ori_data=img, data_name=os.path.basename(path)[:-4])

        # label
        src_label_dir = os.path.join(self.src_root, "label")    # bound, not label
        label_paths = glob.glob(os.path.join(src_label_dir, "*.png"))
        print("\nProcess label...")
        for path in tqdm.tqdm(label_paths):
            src_label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # remove bound
            src_label[src_label == 76] = 0
            src_label[src_label == 149] = 0

            contours, _ = cv2.findContours(src_label, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            label_uint16 = np.zeros(shape=src_label.shape, dtype=np.uint16)
            for i, contour in enumerate(contours):
                tmp_label = np.zeros(shape=src_label.shape, dtype=np.uint8)
                tmp_label = cv2.drawContours(tmp_label, [contour], -1, 255, -1)

                # 膨胀补上边界
                dilation_kernel = np.ones((3, 3), np.uint8)
                tmp_label = cv2.dilate(tmp_label, dilation_kernel, iterations=2)

                tmp_contours, _ = cv2.findContours(tmp_label,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                label_uint16 = cv2.drawContours(label_uint16, tmp_contours, -1,
                                                color=i + 1, thickness=-1)

            self.save_label(ori_label=label_uint16,
                            label_name=os.path.basename(path)[:-4])

        self.save_info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessHistology(args.src_root, args.dst_root,
                        args.dst_size, args.dst_prefix).process()
