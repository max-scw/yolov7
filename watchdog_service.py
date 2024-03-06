import argparse
from pathlib import Path
from typing import Union, List, Tuple

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import cv2 as cv

from detect import Inference

# from utils.general import Log
import logging
from utils.general import set_logging


class Watchmen:
    def __init__(self, directory_to_watch: Path, event_handler, watch_time_interval: int = 1):
        self.directory_to_watch = directory_to_watch.as_posix()
        self.watch_time_interval = int(watch_time_interval)
        self.observer = Observer()

        self.event_handler = event_handler

    def run(self):
        self.observer.schedule(self.event_handler, self.directory_to_watch, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(self.watch_time_interval)
        finally:
            self.observer.stop()
            print("Observer Stopped")
            self.observer.join()


class Handler(FileSystemEventHandler):
    def __init__(
            self,
            path_to_weights: Union[str, Path],
            image_size2display: int = 1020,
            th_score: float = 0.5,
            th_iou: float = 0.5,
            class_colors: List[Tuple[int, int, int]] = None,
            path_to_result_log: Union[str, Path] = None
    ) -> None:
        self.model = Inference(
            weights=path_to_weights,
            imgsz=image_size2display,
            colors=class_colors,
        )

        self.image_size2display = image_size2display
        self.th_score = th_score
        self.th_iou = th_iou
        set_logging(filename=path_to_result_log)
        self._log = logging.getLogger(__name__)
        # initialize log
        self._log.info("New startup...")
        self._log.info(f"#boxes,{','.join(self.model.names)},path")
        # opencv window name
        self.__window_name = "Inference YOLOv7: " + Path(path_to_weights).stem
        print("Eventhandler initialized.")
        self.__imgs_to_del = []

    def on_created(self, event):
        event_path = Path(event.src_path)
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # event_path.unlink()
            for i in range(len(self.__imgs_to_del)):
                p2img = self.__imgs_to_del.pop()
                Path(p2img).unlink()

            print(f"Watchdog received created event - {event_path.as_posix()}")
            if event_path.suffix not in [".bmp", ".jpg", ".png", ".tiff"]:
                return None
            # sleep to allow the system do free the resource
            time.sleep(0.1)

            image_cv = cv.imread(event_path.as_posix(), cv.IMREAD_COLOR)
            # image_cv = cv.resize(image_cv, (640, 640))
            x = self.model.predict(image_cv, conf_thres=self.th_score, iou_thres=self.th_iou)
            self.model.plot_box(image_cv.copy(), *x, show_img=True, bgr2rgb=True)

            # organize images (move them)
            xyxy, scores, classes = x
            if len(classes) == 0:
                # delete image if no box was detected
                self.__imgs_to_del.append(event_path)
            else:
                class_count = ",".join([str(classes.count(i)) for i in range(self.model.n_classes)])
                # log result
                msg = f"{len(classes)},{class_count},{event_path.as_posix()}"
                self._log.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='Path to model weights')
    parser.add_argument('--img-size', type=int, default=640, help='Size of displayed image (in pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.55, help='IOU threshold for NMS')
    parser.add_argument('--folder', type=str, default='ftp',
                        help='Path to folder that should be watched for any changes')
    parser.add_argument('--class-colors', type=str, default='',
                        help='Path to text file that lists the colors for each class.')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Time interval to check for new files in seconds.')

    opt = parser.parse_args()

    print(opt)

    if Path(opt.class_colors).is_file():
        with open(opt.class_colors, "r") as fid:
            lines = fid.readlines()

        class_colors = [ln.strip() for ln in lines]
        print(f"Class colors: {class_colors}")
    else:
        class_colors = None

    event_handler_obj = Handler(
        path_to_weights=opt.weights,
        image_size2display=opt.img_size,
        th_score=opt.conf_thres,
        th_iou=opt.iou_thres,
        class_colors=class_colors,
        path_to_result_log="results.log"
    )
    watch = Watchmen(Path(opt.folder), event_handler_obj, opt.interval)
    print("run watchdog...")
    watch.run()
