import argparse
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from typing import Union, Tuple, List
from utils.debugging import print_debug_msg
from utils.datasets import letterbox


def detect(opt):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    print_debug_msg(f"source={source}, trace={trace}")

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    print_debug_msg(f"imgsz={imgsz}")

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        print_debug_msg(f"conf_thres={opt.conf_thres}, iou_thres={opt.iou_thres}, classes={opt.classes}, agnostic_nms={opt.agnostic_nms}")
        pred2 = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = pred2
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv.imshow(str(p), im0)
                cv.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


class Inference:
    def __init__(self,
                 weights: Union[str, Path],
                 imgsz: int = 640,
                 device_type: str = "cpu",
                 colors: List[List] = None
                 ) -> None:
        # Initialize
        self.device = select_device(device_type)
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        weights = Path(weights).with_suffix(".pt")
        assert weights.exists(), f"Specified weights file '{weights.as_posix()}' does not exist."
        self.model = attempt_load(weights.as_posix(), map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16
        self._warmed_up = False

        self.stride = int(self.model.stride.max())  # grid size (max stride)
        assert self.stride <= 32, f"Maximum stride of the model should be 32 but was {self.stride}."
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        self.__colors = colors

    @property
    def names(self) -> list:
        return self.model.module.names if hasattr(self.model, 'module') else self.model.names

    @property
    def n_classes(self) -> int:
        return len(self.names)

    @property
    def colors(self) -> list:
        if self.__colors is None:
            self.__colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        return self.__colors

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        # Padded resize
        img_pad = letterbox(image, new_shape=self.imgsz, stride=self.stride)[0]
        # Convert
        img_pad = img_pad[:, :, ::-1].transpose(2, 0, 1)  # bring color channels to front # BGR2RGB
        img_pad = np.ascontiguousarray(img_pad)

        img = torch.from_numpy(img_pad).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def _warm_up(self, img: np.ndarray) -> bool:
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        # Warmup
        if self.device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            for _ in range(3):
                self.model(img, augment=False)[0]
        t1 = time.time()
        print(f"Warmup took {(1E3 * (t1 - t0)):.1f}ms.")
        self._warmed_up = True
        return True

    def predict(self,
                image: np.ndarray,
                augment: bool = False,
                conf_thres: float = 0.25,
                iou_thres: float = 0.45,
                agnostic_nms: bool = False,
                classes_to_filter: int = None,
                to_list: bool = True,
                rescale_boxes: bool = True
                ) -> (torch.Tensor, List[float], List[int]):
        img = self._prepare_image(image)

        if not self._warmed_up:
            self._warm_up(img)
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes_to_filter, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        det = pred[0]  # there is only one image
        info = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            if rescale_boxes:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = torch.sum(det[:, -1] == c)  # detections per class
                info.append(f"{n} {self.names[int(c)]}{'s' * (n > 1)}")

        # Print time (inference + NMS)
        print(f'Predictions: {", ".join(info)}. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        xyxy = det[..., :4].int()
        scores = det[..., 4]
        classes = det[..., 5].int()
        if to_list:
            return xyxy.tolist(), scores.tolist(), classes.tolist()
        return xyxy, scores, classes

    def plot_box(self,
                 image: np.ndarray,
                 bbox: Union[torch.Tensor, np.ndarray],
                 scores: List[float],
                 classes: List[int],
                 line_thickness: int = 2,
                 show_img: bool = False,
                 bgr2rgb: bool = False
                 ) -> bool:
        for xyxy, conf, cls in zip(bbox, scores, classes):
            label = f'{self.names[int(cls)]} {conf:.2f}'
            color = self.colors[int(cls)]
            if bgr2rgb:
                color = [color[2], color[1], color[0]]
            plot_one_box(xyxy, image, label=label, color=color, line_thickness=line_thickness)

        if show_img:
            cv.imshow("YOLOv7 Inference", image)
            cv.waitKey(2)  # 1 millisecond
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov7.pt']:
    #             detect(opt)
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect(opt)



    img_size = 640
    stride = 32
    # read file pointing to testing data
    with open("Tst.txt", "r") as fid:
        lines = fid.readlines()
    # convert relevant paths to pathlib object
    images = [Path(ln.strip("\n")) for ln in lines if len(ln) > 5]

    print("____class____")
    mdl = Inference(weights=Path("trained_models") / "2023-03-28_tiny-yolov7-CNN4VIAB640x640-scratch.pt",
                    imgsz=img_size,
                    # colors=[[45, 66, 117], [40, 185, 218]]
                    )
    for p2img in images:
        # read one image
        img = cv.imread(p2img.as_posix(), cv.IMREAD_COLOR)
        x = mdl.predict(img, conf_thres=0.65)

        mdl.plot_box(img.copy(), *x, show_img=True, bgr2rgb=True)
        cv.waitKey(5)  # 1 millisecond
        break
