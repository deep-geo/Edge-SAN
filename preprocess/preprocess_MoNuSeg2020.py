"""
Script for MoNuSeg2020 dataset preprocessing.

Original dataset structure:
.
├── TCGA-55-1594-01Z-00-DX1
│   ├── TCGA-55-1594-01Z-00-DX1_001.svs
│   ├── TCGA-55-1594-01Z-00-DX1_001.tif
│   ├── TCGA-55-1594-01Z-00-DX1_001.xml
│   ├── TCGA-55-1594-01Z-00-DX1_002.svs
│   ├── TCGA-55-1594-01Z-00-DX1_002.tif
│   ├── TCGA-55-1594-01Z-00-DX1_002.xml
│   ...
├── TCGA-5P-A9K0-01Z-00-DX1
│   ├── TCGA-5P-A9K0-01Z-00-DX1_1.svs
│   ├── TCGA-5P-A9K0-01Z-00-DX1_1.tif
│   ├── TCGA-5P-A9K0-01Z-00-DX1_1.xml
│   ├── TCGA-5P-A9K0-01Z-00-DX1_2.svs
│   ├── TCGA-5P-A9K0-01Z-00-DX1_2.tif
│   ├── TCGA-5P-A9K0-01Z-00-DX1_2.xml
│   ...
...

"""
import os
import glob
import argparse
import cv2
import tqdm
import random
import numpy as np
import xml.etree.ElementTree as ET

from preprocess import Preprocess


def parse_aperio_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    # Iterate through all annotations
    for annotation in root.findall('.//Annotation'):
        annotation_data = {
            'id': annotation.get('Id'),
            'type': annotation.get('Type'),
            'name': annotation.get('Name'),
            'regions': []
        }

        # Iterate through all regions in an annotation
        for region in annotation.findall('.//Region'):
            region_data = {
                'id': region.get('Id'),
                'type': region.get('Type'),
                'vertices': []
            }

            # Iterate through all vertices in a region
            for vertex in region.findall('.//Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                region_data['vertices'].append((x, y))

            annotation_data['regions'].append(region_data)

        annotations.append(annotation_data)

    return annotations


def draw_annotations_on_image(annotations, image_size):
    image = np.zeros((image_size[0], image_size[1]), dtype=np.uint16)

    regions = []
    for annotation in annotations:
        for region in annotation["regions"]:
            regions.append(region)

    random.shuffle(regions)
    for i, region in enumerate(regions):
        points = np.array(region['vertices'], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=i + 1, thickness=2)
        cv2.fillPoly(image, [points], color=i + 1)

    return image


class PreprocessMoNuSeg2020(Preprocess):

    def process(self):
        # data
        src_data_paths = glob.glob(os.path.join(self.src_root, "*", "*.tif"))
        print("\nProcess data...")
        for path in tqdm.tqdm(src_data_paths):
            img = cv2.imread(path)
            self.save_data(ori_data=img, data_name=os.path.basename(path)[:-4])

            label_path = path[:-4] + ".xml"
            annotations = parse_aperio_xml(label_path)
            label = draw_annotations_on_image(annotations, img.shape)

            self.save_label(ori_label=label,
                            label_name=os.path.basename(path)[:-4])

        self.save_info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str)
    parser.add_argument("--dst_root", type=str)
    parser.add_argument("--dst_size", type=int)
    parser.add_argument("--dst_prefix", type=str, default="")
    args = parser.parse_args()

    PreprocessMoNuSeg2020(args.src_root, args.dst_root,
                          args.dst_size, args.dst_prefix).process()
