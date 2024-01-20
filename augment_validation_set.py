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
    # strip newline characters from data (and ignore almost empty lines)
    return [el.strip("\n") for el in data if len(el) > 10]


class AugmentFiles:
    __window_name = "AugmentFiles"
    __name_ext = "_aug"

    def __init__(self,
                 path_to_config_data: Union[str, Path],
                 task: str,
                 path_to_albumentations_config: Union[str, Path],
                 oversampling: int = 1,
                 augmentation_probability: float = 0.5,
                 annotation_type="bbox",
                 export_path: Union[str, Path] = "",
                 do_plot: bool = False
                 ):

        self.path_to_data = self._read_data_config(path_to_config_data, task)
        self.oversampling = oversampling
        self.do_plot = do_plot

        # load file with validation data (from data config file)
        data = read_text_file(self.path_to_data)
        # cast to pathlib objects
        self.files = [self.path_to_data.parent / el.strip() for el in data if len(el) > 10]

        # set / initialize augmentation
        self.aug = Albumentations(
            config_file=path_to_albumentations_config,
            p=augmentation_probability,
            annotation_type=annotation_type
        )

        # prepare export
        if not export_path:
            export_path = self.files[0].parent.parent / "aug"
        self._path_to_export = Path(export_path)
        if not self._path_to_export.exists():
            self._path_to_export.mkdir()

    @staticmethod
    def _read_data_config(path_to_config_data: Union[str, Path], task: str) -> Path:
        path_to_config_data = Path(path_to_config_data)

        # load data config file
        with open(path_to_config_data, "r") as fid:
            data_config = yaml.safe_load(fid)
        # path to data file
        if task not in data_config:
            raise ValueError(f"Task '{task}' not found in data config file {path_to_config_data.as_posix()}.")
        return Path(data_config[task])

    @staticmethod
    def _read_label_file(path_to_file: Path) -> np.ndarray:
        labels = read_text_file(path_to_file.with_suffix(".txt"))
        labels = [[float(e) for e in el.split(" ")] for el in labels]
        # convert to numpy.ndarray object
        return np.asarray(labels)

    @staticmethod
    def _draw_labeled_image(img: np.ndarray, label_xywh: np.ndarray) -> np.ndarray:
        img_batch = np.expand_dims(np.moveaxis(img, 2, 0), 0)
        label_batch = np.hstack((np.zeros((label_xywh.shape[0], 1)), label_xywh))
        return plot_images(img_batch, label_batch, max_size=max(img.shape))

    def show_labeled_image(self, img: np.ndarray, label_xywh: np.ndarray,
                           img_ogl: np.ndarray, label_xywh_ogl: np.ndarray):

        img1 = self._draw_labeled_image(img_ogl, label_xywh_ogl)
        img2 = self._draw_labeled_image(img, label_xywh)
        img_cv = np.hstack((img1, img2))
        cv.imshow(self.__window_name, img_cv)
        cv.waitKey(0)

    def augment_files(self):
        info = []
        for p2fl in self.files:
            # keep track of files
            info.append(p2fl)
            # read image
            img = cv.imread(p2fl.as_posix(), cv.IMREAD_COLOR)
            # read labels
            labels = self._read_label_file(p2fl)
            # to absolute coordinates
            labels[..., 1:] *= (img.shape[:2][::-1] * 2)
            # to corner coordinates
            xyxy = np.hstack((labels[..., 0].reshape(-1, 1), xywh2xyxy(labels[..., 1:])))

            for i in range(self.oversampling):
                # augment image + labels
                img_augmented, labels_augmented, mask_augmented = self.aug(img, xyxy)
                # img_augmented, labels_augmented = img, xyxy
                labels_augmented_xywh = np.hstack(
                    (labels_augmented[..., 0].reshape(-1, 1), xyxy2xywh(labels_augmented[..., 1:])))

                # write augmented image
                filename = p2fl.stem + self.__name_ext + p2fl.suffix
                path_to_file_new = self._path_to_export / filename
                cv.imwrite(path_to_file_new.as_posix(), img_augmented)
                # write augmented labels
                labels_augmented_xywh /= ((1,) + (img_augmented.shape[:2][::-1] * 2))
                with open(path_to_file_new.with_suffix(".txt"), "w") as fid:
                    for row in labels_augmented_xywh:
                        fid.write(" ".join([f"{el:.5}" for el in row]) + "\n")

                # keep track of files
                info.append(path_to_file_new)

                if self.do_plot:
                    # TODO: stack with original image
                    self.show_labeled_image(img_augmented, labels_augmented_xywh, img, labels)
                    break
        # close window if necessary
        if self.do_plot:
            cv.destroyWindow(self.__window_name)

        # write new data file
        path_to_data_val_new = self.path_to_data.with_stem(self.path_to_data.stem + self.__name_ext)
        with open(path_to_data_val_new, "w") as fid:
            for el in info:
                fid.write(el.as_posix() + "\n")

        print(f"Created {int(len(info) / 2)} new files. Paths written to {path_to_data_val_new.as_posix()}.")


if __name__ == "__main__":

    AugmentFiles(
        path_to_config_data="data/VIAB1.yaml",
        task="val",
        path_to_albumentations_config="data/augmentation-VIAB.yaml",
        augmentation_probability=0.2,
        oversampling=2,
        do_plot=False
    ).augment_files()
