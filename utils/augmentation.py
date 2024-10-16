from pathlib import Path
import yaml
import re
from ast import literal_eval

import albumentations as A
import albumentations  # for eval()

import cv2  # for eval()
import cv2 as cv  # for eval()

import numpy as np

from typing import Union, Dict

from utils.plots import plot_images, plot_one_box
from utils.general import xywh2xyxy


def build_augmentation_pipeline(
        config_file: Union[str, Path],
        probability: float = None,
        verbose: bool = False,
        overwrite_probability: bool = False
) -> list:

    config_file = Path(config_file)
    if not config_file.exists() or not config_file.is_file():
        raise FileExistsError(f"Configuration file {config_file} does not exist.")

    if config_file.suffix.lower() not in [".yaml", ".yml"]:
        raise Exception(f"Expecting a YAML file but the extension of the provided config file was {config_file.suffix}.")

    # read augmentation config
    with open(config_file, "r") as fid:
        config_txt = yaml.load(fid, Loader=yaml.SafeLoader)

    # parse / cast function arguments
    config = parse_arguments(config_txt)
    # if verbose:
    #     print(f"Augmentation specs: {config}")

    # create pipelines:
    # TODO: support other transformation packages
    transformations = []
    if "albumentations" in config:
        transformations = build_albumentations_pipeline(
            config["albumentations"],
            p=probability,
            verbose=verbose,
            overwrite_p=overwrite_probability,
        )
    return transformations


def parse_arguments(config: Dict[str, dict]) -> Dict[str, dict]:
    """
    Parses the dictionary from a YAML file
    NOTE: recursive function
    :param config: dictionary (potentially of dictionaries) of strings
    :return: dictionary of the same depth but with cast values (where applicable) and empty dictionaries where no arguments were provided.
    """
    # prepare regex patterns for casting
    pattern_float = "(\-?\d+\.\d*|\.\d+)"
    pattern_int = "\-?\d+"
    pattern_number = f"({pattern_float}|{pattern_int})"

    # tuple
    re_numeric_tuple = re.compile(f"\((\s?{pattern_number}\s?,\s?)+{pattern_number}?\s?\)")
    # re_generic_tuple = re.compile(r"^\([^)]*\)$")

    # list
    re_numeric_list = re.compile(f"\[(\s?{pattern_number}\s?,?\s?)+\]")

    # boolean
    re_bool = re.compile("(False|True)", re.IGNORECASE)
    re_true = re.compile("True", re.IGNORECASE)

    # flags
    re_flags_cv2 = re.compile("cv2?\.[A-Z][A-Z0-9_]+")
    re_flags_albumentations = re.compile("(A|albumentations)(\.[A-Z][A-Za-z]+)+")

    # loop through transformation packages
    config_ = dict()
    for ky, val in config.items():
        # if further arguments are specified
        if isinstance(val, dict):
            # recursive call
            val = parse_arguments(val)
        elif isinstance(val, str):
            # cast string input if applicable
            if re_numeric_tuple.match(val) or re_numeric_list.match(val):
                val = literal_eval(val)
            elif re_bool.match(val):
                val = True if re_true.match(val) else False
            elif re_flags_cv2.match(val) or re_flags_albumentations.match(val):
                # ensure that the eval() output wont expose a security risk by enforcing an integer return
                val = int(eval(val))
        elif not val:
            val = dict()
        config_[ky] = val
    return config_


def build_albumentations_pipeline(
        config: Dict[str, dict],
        p: float = None,
        verbose: bool = False,
        overwrite_p: bool = False,
) -> list:
    """
    creates a list of transformation functions from the albumentations package
    :param config: Dictionary specifying the function names and arguments of each function
    :param p: probability to apply a transformation. This value overwrites the values in the config file
    :param verbose: prints the transformation function
    :param overwrite_p: overwrites the values in the config file
    :return: list of albumentations transformation functions
    """

    # loop through specified functions and options
    trafo_fncs = []
    for fnc, opts in config.items():

        if p and (overwrite_p or ("p" not in opts)):
            opts["p"] = p

        # get transformation function
        transformation = getattr(A, fnc)(**opts)
        if verbose:
            # logging.info(colorstr('albumentations: ') + str())
            print(transformation)
        trafo_fncs.append(transformation)
    return trafo_fncs


def create_examples(
        config_file: Union[str, Path],
        image_dir: Union[str, Path],
        image_ext: str = "*",
        export_dir: Union[str, Path] = None,
        oversampling: int = 1
):
    trafo_fnc = build_augmentation_pipeline(
        config_file,
        verbose=True
    )

    transforms = dict()
    for fnc in trafo_fnc:
        # always apply function
        fnc.always_apply = True
        # get class name
        name = fnc.__class__.__name__
        # compose
        kwargs = {"bbox_params": A.BboxParams(format="yolo", label_fields=["class_labels"])}
        transforms[name] = A.Compose([fnc], **kwargs)

    # get images that should be transformed
    image_dir_ = Path(image_dir)
    if image_dir_.is_file():
        images = [image_dir_]
    elif image_dir_.is_dir():
        images = list(image_dir_.glob(f"**/*.{image_ext}"))

    export_dir = Path(export_dir)

    # apply transform
    n_col = max((int(round(np.sqrt(oversampling * (16 / 9)))), 1))
    for p2img in images:
        # read image
        img = cv.imread(p2img.as_posix(), cv.IMREAD_COLOR)
        # read label
        if p2img.with_suffix(".txt").exists():
            lbl = np.loadtxt(p2img.with_suffix(".txt"))
        else:
            lbl = None

        # apply trafos
        for nm, fnc in transforms.items():
            print(f"{nm}: {fnc}")
            imgs_oversampled, row = [], []
            j = 0
            for i in range(oversampling):
                out = fnc(image=img.copy(), bboxes=lbl[:, 1:5], class_labels=lbl[:, 0])
                img_transformed = out["image"]
                bbx_transformed = out["bboxes"]

                bbx_xyxy = xywh2xyxy(np.asarray(bbx_transformed)) * (img_transformed.shape[:2][::-1] * 2)
                for bbx in bbx_xyxy.astype(int):
                    plot_one_box(bbx, img_transformed, (0, 255, 0))
                # store transformed image
                row.append(img_transformed)

                j += 1
                if j >= n_col:
                    # pad images if necessary
                    max_shape = np.array([el.shape for el in row]).max(axis=0)
                    row_pad = [np.pad(el, [(0, x) for x in max_shape - el.shape], mode='constant') for el in row]
                    imgs_oversampled.append(np.hstack(row_pad))

                    row = []
                    j = 0
                elif i >= (oversampling - 1):
                    row_mat = np.hstack(row)
                    if len(imgs_oversampled) > 0:
                        # padding
                        row_pad = np.zeros((row_mat.shape[0], imgs_oversampled[0].shape[1] - row_mat.shape[1], row_mat.shape[2]))
                        row_mat = np.hstack([row_mat, row_pad])
                    imgs_oversampled.append(row_mat)
            # stack images
            img_stacked_transforms = np.vstack(imgs_oversampled)


            p2save = export_dir / nm
            if not p2save.exists():
                p2save.mkdir(parents=True)

            # write image to folder
            filename = p2save / f"{p2img.stem}_{nm}{p2img.suffix}"
            cv.imwrite(filename.as_posix(), img_stacked_transforms)
    return True


if __name__ == "__main__":
    # build_augmentation_pipeline("../data/Test-augmentation.yaml", probability=0.1)

    create_examples(
        config_file="../data/Test-augmentation.yaml",
        image_dir=r"..\data\img\2024-06-20_19-34-42-6930jfz_Input0_Camera0.jpg",
        export_dir="export",
        oversampling=16
    )
