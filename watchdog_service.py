import argparse
import pathlib as pl
from typing import Union, List

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import cv2 as cv

from detect import Inference


class Watchmen:
    def __init__(self, directory_to_watch: pl.Path, event_handler, watch_time_interval: int = 1):
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
    def __init__(self,
                 path_to_weights: Union[str, pl.Path],
                 image_size2display: int = 1020,
                 th_score: float = 0.5,
                 th_iou: float = 0.5,
                 class_colors: List[list] = None
                 ):
        self.model = Inference(weights=path_to_weights,
                               imgsz=image_size2display,
                               colors=class_colors,
                               )

        self.image_size2display = image_size2display
        self.th_score = th_score
        self.th_iou = th_iou
        # opencv window name
        self.window_name = "Inference YOLOv7: " + pl.Path(path_to_weights).stem
        print("Eventhandler initialized.")

    def on_created(self, event):
        event_path = pl.Path(event.src_path)
        if event.is_directory:
            return None

        elif event.event_type == 'created':

            print(f"Watchdog received created event - {event_path.as_posix()}")
            if event_path.suffix not in [".bmp", ".jpg", ".png"]:
                return None
            # sleep to allow the system do free the resource
            time.sleep(0.1)

            image_cv = cv.imread(event_path.as_posix(), cv.IMREAD_COLOR)
            x = self.model.predict(image_cv, conf_thres=self.th_score, iou_thres=self.th_iou)
            self.model.plot_box(image_cv.copy(), *x, show_img=True, bgr2rgb=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='size of displayed image (in pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    opt = parser.parse_args()
    print(opt)

    folder_to_monitor = pl.Path(r"C:\Users\schwmax\Downloads\FTP_ENTRYPOINT")
    time_interval_s = 1  # second

    event_handler_obj = Handler(path_to_weights=opt.weights,
                                image_size2display=opt.img_size,
                                th_score=opt.conf_thres,
                                th_iou=opt.iou_thres,
                                class_colors=[[45, 66, 117], [40, 185, 218]]
                                )
    watch = Watchmen(folder_to_monitor, event_handler_obj, time_interval_s)
    print("run watchdog...")
    watch.run()
