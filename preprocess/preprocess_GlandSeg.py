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
import tqdm

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
            self.save_data(ori_data=img, data_name=f"image_{i + 1:04d}")

        # label
        print("\nProcess label...")
        for i, path in tqdm.tqdm(enumerate(src_label_paths),
                                 total=len(src_label_paths)):
            label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.save_label(ori_label=label, label_name=f"image_{i + 1:04d}")

        self.save_info(info_data={"count": self.count})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessGlandSeg(args.src_root, args.dst_root,
                       args.dst_size, args.dst_prefix).process()

