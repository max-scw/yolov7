from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2 as cv
from typing import Union, Tuple, List


def make_video_from_files(path_to_images: Path,
                          file_name: Union[str, Path],
                          frame_size: Union[int, Tuple[int, int]] = 1920,
                          frames_per_second: int = 30,
                          image_extension: str = "*",
                          alpha_channel: bool = False
                          ) -> bool:
    images = list(path_to_images.glob(f"*.{image_extension.strip('.')}"))
    if isinstance(frame_size, int):
        img = cv.imread(images[0].as_posix(), cv.IMREAD_UNCHANGED)  # IMREAD_GRAYSCALE
        frame_size = shape_to_max_size(img.shape[:2], frame_size)[::-1]

    # FourCC video format
    if alpha_channel:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        file_extension = ".mp4"
    else:
        fourcc = cv.VideoWriter_fourcc(*"DIVX")
        file_extension = ".avi"

    # Create the video writer object
    file_path = Path(file_name).with_suffix(file_extension).as_posix()
    if alpha_channel:
        raise ValueError("Transparency option does not exist for OpenCV's Python version.")
    else:
        video = cv.VideoWriter(file_path, fourcc, frames_per_second, frame_size)

    # Loop through each image and write it to the video
    i = 0
    for i, p2img in enumerate(images):
        # load image
        img = cv.imread(p2img.as_posix(), cv.IMREAD_UNCHANGED if alpha_channel else cv.IMREAD_COLOR)  # IMREAD_GRAYSCALE
        # resize image
        img_sml = cv.resize(img, dsize=frame_size, interpolation=cv.INTER_CUBIC)
        # add image to video
        video.write(img_sml)
    video.release()

    print(f"{i} images written to {file_path}.")
    return True


def loss_to_images(path_to_results: Union[str, Path],
                   export_dir: Union[str, Path] = "tmp",
                   fig_size: Tuple[int, int] = (1920, 1080),
                   color: str = "#28B9DA",
                   max_epoch: int = -1,
                   step: int = 1
                   ):
    with open(path_to_results) as fid:
        text = fid.readlines()

    text_lines = []
    for line in text:
        text_lines.append([el for el in line.strip("\n").split(" ") if el != ""])
    numbers = []
    for line in text_lines:
        numbers.append([float(el) for el in line[2:]])
    results = np.array(numbers)

    if max_epoch < 1:
        max_epoch = results.shape[0]

    n_rows = 3
    n_cols = int(np.ceil(results.shape[1] / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols)

    tags = ["train/box_loss", "train/obj_loss", "train/cls_loss",  # train loss
            "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95",
            "val/box_loss", "val/obj_loss", "val/cls_loss",  # val loss
            "x/lr0", "x/lr1", "x/lr2"]

    for i, ax in enumerate(axs.flatten()):
        if i >= results.shape[1]:
            break
        ax.plot(results[:, i])
        ax.set_ylabel(tags[i])
    fig.tight_layout()
    fig.show()
    idx = 8

    path_to_export_dir = Path(export_dir)
    if not path_to_export_dir.exists():
        path_to_export_dir.mkdir()

    frame_size_fig = tuple([int(el/200) for el in fig_size])

    for i in range(0, max_epoch, step):
        fig = plt.figure(figsize=frame_size_fig, dpi=300)
        canvas = FigureCanvas(fig)
        plt.plot(results[:i, idx], color=color, linewidth=1)
        ax = fig.gca()
        # ax.xlabel("epoch")
        # ax.ylabel("loss")
        ax.tick_params(left=False, right=False,
                       labelleft=False, labelbottom=False,
                       bottom=False)
        ax.set_xlim([0, round(max_epoch * 1.05)])
        ax.set_ylim([0, 1.25 * results[:, idx].max()])
        ax.set_facecolor("black")
        fig.set_facecolor("black")  # the background of the axis is the foreground of the figure
        for ky in ["top", "bottom", "left", "right"]:
            ax.spines[ky].set_color("white")
            ax.spines[ky].set_linewidth(3)

        canvas.draw()  # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi()
        array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plt.close(fig)
        mat = array.reshape(int(height), int(width), 3)
        # add alpha channel
        rgba = cv.cvtColor(mat, cv.COLOR_RGB2RGBA)
        # create alpha channel (set white to be transparent (i.e. 255 in the alpha channel)
        alpha = (mat.sum(2) / 3 > 50) * 255
        rgba[..., 3] = alpha
        img = cv.cvtColor(rgba, cv.COLOR_RGBA2BGRA)

        filename = f"loss_{str(i).zfill(len(str(results.shape[0])))}.png"
        export_path = path_to_export_dir / filename
        cv.imwrite(export_path.as_posix(), img)


def shape_to_max_size(imgsz, max_image_size: int) -> Tuple[int, int]:
    scale_factor = max_image_size / max(imgsz)
    if scale_factor < 1:
        imgsz = tuple(np.ceil(scale_factor * np.array(imgsz)).astype(int))
    return imgsz


def input_to_images(images: List[Union[str, Path]],
                    path_to_export: Union[str, Path],
                    max_image_size: int = 200,
                    aspect_ratio: float = 16 / 9,
                    monochrome: bool = False
                    ):
    n_images = len(images)

    n_rows = int(np.round(np.sqrt(n_images / aspect_ratio)))
    n_cols = int(np.ceil(n_images / n_rows))

    for i, p2fl in enumerate(images):
        img = cv.imread(p2fl.as_posix(), cv.IMREAD_GRAYSCALE if monochrome else cv.IMREAD_COLOR)

        h, w = shape_to_max_size(img.shape[:2], max_image_size)
        img = cv.resize(img, (w, h))

        if i == 0:
            # initialize mosaic image
            mosaic_sz = (int(n_rows * h), int(n_cols * w))
            if not monochrome:
                mosaic_sz += (3,)
            mosaic = np.full(mosaic_sz, 255, dtype=np.uint8)  # init

        i_x = i % n_cols
        i_y = i // n_cols
        block_x = int(w * i_x)
        block_y = mosaic.shape[0] - int(h * i_y)

        mosaic[block_y - h:block_y, block_x:block_x + w, ...] = img

        cv.rectangle(mosaic, (block_x, block_y - h), (block_x + w, block_y), 255, thickness=10)

        # write image
        if not path_to_export.exists():
            path_to_export.mkdir()
        filename = f"Overview_{str(i).zfill(len(str(n_images)))}.jpg"
        cv.imwrite((path_to_export / filename).as_posix(), mosaic)
        if i > n_rows * n_cols:
            break


if __name__ == "__main__":

    max_size = 1920

    # # ---- loss images
    path_to_result_file = Path("results.txt")
    path_to_dir_loss = Path("loss")
    # loss_to_images(path_to_result_file,
    #                path_to_dir_loss,
    #                fig_size=(1280, 720),
    #                # max_epoch=5,
    #                step=30)
    make_video_from_files(path_to_images=path_to_dir_loss,
                          file_name="loss_3",
                          frame_size=max_size,
                          frames_per_second=80,
                          )

    # # ---- input images
    # path_to_input_images = Path("../dataset/NEW/img")
    # path_to_export_overview = Path("tmp3")
    #
    # # input_to_images(list(path_to_input_images.glob("*.bmp")),
    # #                 path_to_export_overview,
    # #                 monochrome=True)
    #
    # make_video_from_files(path_to_images=path_to_export_overview,
    #                       file_name="Overview.avi",
    #                       frame_size=max_size,
    #                       frames_per_second=23,
    #                       )

    # ---- augmented images
    # path_to_augmented_fir = Path("TEST_tiny-yolo7")
    # make_video_from_files(path_to_images=path_to_augmented_fir,
    #                       file_name="augmented_3.avi",
    #                       frame_size=max_size,
    #                       frames_per_second=10,
    #                       )
    print("done.")
