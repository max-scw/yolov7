import numpy as np
from PIL import Image
import cv2 as cv
from pathlib import Path
import itertools
from random import shuffle

from typing import Tuple

from utils.plots import plot_one_box
from utils.general import xywh2xyxy


def draw_keypoint(img: np.ndarray,
                  xy: Tuple[int, int],
                  color_range: Tuple[int, int],
                  mode: int
                  ) -> None:
    color = int(np.random.randint(color_range[0], color_range[1], 1))
    if mode == 0:
        # circle
        sz = 5
        # mat[xy[0] - sz:xy[0] + sz + 1, xy[1]] = color(sz * 2 + 1)
        # mat[xy[0], xy[1] - sz:xy[1] + sz + 1] = color(sz * 2 + 1)
        cv.circle(img, xy, sz, (color,), -1)
    elif mode == 1:
        # box
        sz = 4
        cv.rectangle(img, xy - sz, xy + sz, (color,), -1)


if __name__ == "__main__":
    path_export = Path("../dataset/kpt_toy_data")
    image_sz = (640, 640)
    n_images = 100
    background_noise_scale = 50

    n_objects_max = 4
    n_object_classes = 1
    object_sz_max = (100, 100)
    object_aspect_ratio = 4/3
    object_color = (230, 255)

    n_keypoints_per_object = 2
    n_keypoints = 2
    keypoint_color = ((100, 110), (50, 70))

    np.random.seed(42)
    for i_img in range(n_images):

        # add random number of objects in this image
        n_objects = np.random.randint(1, n_objects_max)
        # random object size
        obj_scl = np.random.random((n_objects, len(object_sz_max)))
        obj_scl_w = obj_scl[..., 0]
        obj_scl_h = obj_scl_w / (object_aspect_ratio + (np.random.random((n_objects,)) - 0.5))
        obj_scl = np.hstack((obj_scl_w.reshape(-1, 1), obj_scl_h.reshape(-1, 1))) + 0.1
        # saturate at 10%
        obj_scl = obj_scl.clip(min=0.15, max=1)
        # absolute object size (width, height)
        obj_sz = object_sz_max * obj_scl

        # create random center positions of objects
        x0y0 = np.random.random(obj_sz.shape) * image_sz
        x0y0 = 0.8 * x0y0 + 0.1 * np.asarray(image_sz)
        # x0y0 = x0y0.clip(obj_sz / 2, image_sz - obj_sz / 2)
        # corner coordinates
        xyxy = np.hstack((x0y0, x0y0)) + np.hstack((-obj_sz / 2, obj_sz / 2))
        xyxy = xyxy.astype(int)

        # create random positions of key points within the objects
        xy_kpt_rel = np.random.random((n_objects, n_keypoints_per_object * n_keypoints))
        xy_kpt_rel = 0.9 * xy_kpt_rel + 0.05
        # turn into absolute coordinates
        xy_kpt = np.hstack((xyxy[..., :2],) * n_keypoints_per_object) + \
                 xy_kpt_rel * np.hstack((obj_sz,) * n_keypoints_per_object)
        xy_kpt = xy_kpt.astype(int)

        # create blank image
        img = (np.random.random(image_sz) * background_noise_scale).astype(np.uint8)
        # draw objects
        for obj in xyxy:
            size = (obj[2] - obj[0], obj[3] - obj[1])
            color = int(np.random.randint(object_color[0], object_color[1], size=1))
            cv.rectangle(img, obj[:2], obj[2:], color, -1)
            # img[obj[0]:obj[2], obj[1]:obj[3]] = np.random.randint(object_color[0], object_color[1], size=size)
        # draw key points
        for row in xy_kpt:
            for j, kpt in enumerate(row.reshape(-1, 2)):
                draw_keypoint(img, kpt, keypoint_color[j], j)

        # write image
        path_to_file = path_export / f"IMAGE_{i_img:03}"
        Image.fromarray(img).save(path_to_file.with_suffix(".jpg"))
        # Image.fromarray(img).show()

        # write label file
        with open(path_to_file.with_suffix(".txt"), "w") as fid:
            for obj, row in zip(xyxy, xy_kpt):
                # scale to relative coordinates
                xy1 = obj[..., :2]
                xy2 = obj[..., 2:]
                wh = xy2 - xy1
                xy0 = xy1 + wh / 2
                xywh = np.hstack((xy0, wh))
                obj_rel = xywh / (image_sz * 2)

                row_rel = row / (image_sz * n_keypoints_per_object)

                # obj_rel[1] = 1 - obj_rel[1]
                # row_rel = np.ones_like(row_rel) - row_rel
                # add visibility key to key points (e.g. "2" == "visible)
                kpts_rel = list(itertools.chain(*[list(el) + [2] for el in row_rel.reshape(-1, 2)]))
                # all data that should be stored ([0] is the class label
                info = [0] + list(obj_rel) + list(kpts_rel)
                fid.write(" ".join([str(round(el, 3)) for el in info]) + "\n")

    # ---------- DRAW DATA
    image_paths = list(path_export.glob("*.jpg"))
    #
    for p2fl in image_paths:
        img = Image.open(p2fl)
        # target =
        # read file with labels
        with open(p2fl.with_suffix(".txt"), 'r') as fid:
            lines = [x.split() for x in fid.read().strip().splitlines()]
        # cast to floats (classes are supposed to be integers though)
        lines = np.array(lines, dtype=np.float32)

        img_cv = np.asarray(img)
        xywh = np.asarray(lines).copy()[..., 1:5]
        xyxy = xywh2xyxy(xywh) * (img.size * 2)
        for row in xyxy:
            plot_one_box(row, img_cv)
        Image.fromarray(img_cv).show()
        break

    # ---------- SPLIT DATA
    # distribute images / split data
    # images = list(path_export.glob("*.jpg"))
    split_fraction = {"Trn_kpt_toy": 0.3, "Val_kpt_toy": 0.2, "Tst_kpt_toy": 0.5}

    shuffle(image_paths)
    k = 0
    for ky, val in split_fraction.items():
        n_images = round(len(image_paths) * val)
        with open(f"{ky}.txt", "w") as fid:
            for i in range(n_images):
                fid.write(Path(image_paths[k]).as_posix() + "\n")
                k += 1
        print(f"done {ky}.txt with {n_images} images.")

