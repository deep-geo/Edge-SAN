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
import numpy as np
import tqdm

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
                filename = filenames[i]
                filename = os.path.basename(filename)[:-4]
                if filename not in counter:
                    counter[filename] = 1
                else:
                    counter[filename] += 1

                basename = f"{filename}_{counter[filename]}"

                # data
                img_16bit = X[i, :, :, 0]
                img_normalized = ((img_16bit - img_16bit.min()) /
                                  (img_16bit.max() - img_16bit.min()))
                img = (img_normalized * 255).astype(np.uint8)
                self.save_data(data=img, data_name=basename)

                # label
                label = y[i, :, :, 0]
                self.save_label(label=label, label_name=basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessDynamicNuclearNet(args.src_root, args.dst_root,
                                args.dst_size, args.dst_prefix).process()
