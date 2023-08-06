from pathlib import Path
import yaml
from typing import Union, List
import numpy as np

import cv2 as cv

from utils.datasets import Albumentations
from utils.plots import plot_images
from utils.general import xyxy2xywh, xywh2xyxy


def read_text_file(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"File {path.as_posix()} does not exist.")

    with open(path, "r") as fid:
        data = fid.readlines()
    #
    return [el.strip("\n") for el in data if len(el) > 10]


if __name__ == "__main__":
    path_to_data = Path("data/CNN4VIAB.yaml")


    # load data config file
    with open(path_to_data, "r") as fid:
        data_config = yaml.safe_load(fid)

    # load file with validation data (from data config file)
    path_to_data_val = Path(data_config["val"])
    data = read_text_file(path_to_data_val)
    # cast to pathlib objects
    files = [Path(el.strip("\n")) for el in data if len(el) > 10]

    aug = Albumentations(augment_ogl=False, augmentation_probability=0.5)

    for p2fl in files:
        # read image
        img = cv.imread(p2fl.as_posix(), cv.IMREAD_COLOR)
        # read labels
        labels = read_text_file(p2fl.with_suffix(".txt"))
        labels = [[float(e) for e in el.split(" ")] for el in labels]
        # convert to numpy.ndarray object
        labels = np.asarray(labels)
        # to absolute coordinates
        labels[..., 1:] *= (img.shape[:2][::-1] * 2)
        # to corner coordinates
        xyxy = np.hstack((labels[..., 0].reshape(-1, 1), xywh2xyxy(labels[..., 1:])))

        # augment image + labels
        img_augmented, labels_augmented = aug(img, xyxy)
        # img_augmented, labels_augmented = img, xyxy
        labels_augmented_xywh = np.hstack((labels_augmented[..., 0].reshape(-1, 1), xyxy2xywh(labels_augmented[..., 1:])))

        img_cv = plot_images(np.expand_dims(np.moveaxis(img_augmented, 2, 0), 0),
                             np.hstack((np.zeros((labels_augmented.shape[0], 1)), labels_augmented_xywh)),
                             max_size=max(img.shape))

        # write
        path_to_export = p2fl.parent.parent / "aug"
        if not path_to_export.exists():
            path_to_export.mkdir()

        filename = p2fl.stem + "_aug" + p2fl.suffix
        path_to_file = path_to_export / filename
        cv.imwrite(path_to_file.as_posix(), img_augmented)

        labels_augmented_xywh /= ((1,) + (img_augmented.shape[:2][::-1] * 2))
        with open(path_to_file.with_suffix(".txt"), "w") as fid:
            for row in labels_augmented_xywh:
                fid.write(" ".join([f"{el:.5}" for el in row]) + "\n")

        # cv.imshow("tmp", img_cv)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
