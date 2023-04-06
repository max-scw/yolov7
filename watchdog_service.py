import argparse
from pathlib import Path
from typing import Union, List, Tuple

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import cv2 as cv

from detect import Inference

from utils.general import Log


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
    def __init__(self,
                 path_to_weights: Union[str, Path],
                 image_size2display: int = 1020,
                 th_score: float = 0.5,
                 th_iou: float = 0.5,
                 class_colors: List[Tuple[int, int, int]] = None,
                 path_to_result_log: Union[str, Path] = None
                 ):
        self.model = Inference(weights=path_to_weights,
                               imgsz=image_size2display,
                               colors=class_colors,
                               )

        self.image_size2display = image_size2display
        self.th_score = th_score
        self.th_iou = th_iou
        self._log = Log(path_to_result_log)
        # initialize log
        self._log.log("New startup...")
        self._log.log(f"#boxes,{','.join(self.model.names)},path")
        # opencv window name
        self.window_name = "Inference YOLOv7: " + Path(path_to_weights).stem
        print("Eventhandler initialized.")

    def on_created(self, event):
        event_path = Path(event.src_path)
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            print(f"Watchdog received created event - {event_path.as_posix()}")
            if event_path.suffix not in [".bmp", ".jpg", ".png"]:
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
                event_path.unlink()
            else:
                class_count = ",".join([str(classes.count(i)) for i in range(self.model.n_classes)])
                # log result
                msg = f"{len(classes)},{class_count},{event_path.as_posix()}"
                self._log.log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='size of displayed image (in pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    opt = parser.parse_args()
    opt.weights = Path("trained_models") / "2023-03-28_tiny-yolov7-CNN4VIAB640x640-scratch.pt"
    print(opt)

    folder_to_monitor = Path(r"C:\Users\schwmax\Downloads\FTP_ENTRYPOINT")
    time_interval_s = 1  # second

    event_handler_obj = Handler(path_to_weights=opt.weights,
                                image_size2display=opt.img_size,
                                th_score=opt.conf_thres,
                                th_iou=opt.iou_thres,
                                class_colors=[(137, 186, 23)] + [(255, 0, 0)] * 3, # (192, 0, 0)
                                path_to_result_log="results.log"
                                )
    watch = Watchmen(folder_to_monitor, event_handler_obj, time_interval_s)
    print("run watchdog...")
    watch.run()
