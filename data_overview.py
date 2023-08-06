import pathlib as pl

import numpy as np
import cv2 as cv
from utils.plots import plot_images

import math


if __name__ == "__main__":
    path_to_dataset_info = pl.Path("Trn.txt")

    num_img_per_plot = 15

    # read file with paths to data for this dataset
    with open(path_to_dataset_info, "r") as fid:
        data_info = fid.readlines()
    # convert lines to path objects
    data_info = [pl.Path(ln.strip("\n")) for ln in data_info if len(ln) > 5]

    # only keep to which also a label file exists
    data_info = [el for el in data_info if el.with_suffix(".txt").is_file()]

    num_plots = math.ceil(len(data_info) / num_img_per_plot)
    # [data_info[num_img_per_plot * i:num_img_per_plot * (i + 1)] for i in range(num_plots)]
    for i in range(num_plots):
        idx = num_img_per_plot * i
        paths_to_images = data_info[idx:min(idx + num_img_per_plot, len(data_info))]

        # aggregate images and labels
        images = []
        targets = []
        for j, p2img in enumerate(paths_to_images):
            # read image
            img = cv.imread(p2img.as_posix(), cv.IMREAD_COLOR)
            img = img[:, :, ::-1]  # BGR to RGB
            img = img.transpose(2, 0, 1)  # to 3x416x416

            # append / store images
            images.append(img)

            # read label
            with open(p2img.with_suffix(".txt"), "r") as fid:
                labels_txt = fid.readlines()
            labels = [[j] + ln.strip("\n").split(" ") for ln in labels_txt if len(ln) > 5]
            # cast
            # labels = [(int(c), float(x), float(y), float(w), float(h)) for c, x, y, w, h in labels]
            labels = np.array(labels, dtype=float)
            # append / store labels
            targets.append(labels)

        mosaic = plot_images(np.asarray(images), np.concatenate(targets), max_subplots=num_img_per_plot, max_size=1024, fname=None)
        filename = path_to_dataset_info.stem + f"_mosaic_{i}.jpg"
        cv.imwrite(filename, mosaic)
        print(f"{filename} done.")


