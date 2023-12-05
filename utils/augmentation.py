from pathlib import Path
import yaml
import re
from ast import literal_eval

import albumentations as A
import albumentations  # for eval()

import cv2  # for eval()
import cv2 as cv  # for eval()

from typing import Union, Dict


def build_augmentation_pipeline(config_file: Union[str, Path], probability: float = None, verbose: bool = False):

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
    if verbose:
        print(f"Augmentation specs: {config}")

    # create pipelines:
    # TODO: support other transformation packages
    transformations = []
    if "albumentations" in config:
        transformations = build_albumentations_pipeline(config["albumentations"], p=probability, verbose=verbose)
    return transformations


def parse_arguments(config: Dict[str, dict]) -> Dict[str, dict]:
    """
    Parses the dictionary from a YAML file
    NOTE: recursive function
    :param config: dictionary (potentially of dictionaries) of strings
    :return: dictionary of the same depth but with cast values (where applicable) and empty dictionaries where no arguments were provided.
    """
    # prepare regex patterns for casting
    pattern_float = "(\d+\.\d*|\.\d+)"
    pattern_int = "\d+"
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
        verbose: bool = False) -> list:
    """
    creates a list of transformation functions from the albumentations package
    :param config: Dictionary specifying the function names and arguments of each function
    :param p: probability to apply a transformation. This value overwrites the values in the config file
    :param verbose: prints the transformation function
    :return: list of albumentations transformation functions
    """

    # loop through specified functions and options
    trafo_fncs = []
    for fnc, opts in config.items():
        if p:
            opts["p"] = p

        # get transformation function
        transformation = getattr(A, fnc)(**opts)
        if verbose:
            print(transformation)
        trafo_fncs.append(transformation)
    return trafo_fncs


if __name__ == "__main__":
    build_augmentation_pipeline("../cfg/training/data-augmentation.yaml")