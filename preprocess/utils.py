import cv2
import albumentations as A


def get_transform(img_size, ori_h, ori_w):
    if ori_h < img_size and ori_w < img_size:
        t = A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            position="center", border_mode=cv2.BORDER_CONSTANT,
            value=0
        )
    else:
        t = A.Resize(
            int(img_size), int(img_size),
            interpolation=cv2.INTER_NEAREST  # Other interpolation methods may result in label inaccuracies.
        )
    return t
