# Dataset utils and dataloaders
import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import (
    check_requirements,
    xyxy2xywh,
    xywh2xyxy,
    xywhn2xyxy,
    xyn2xy,
    segment2box,
    segments2boxes,
    resample_segments,
    clean_str,
    colorstr
)
from utils.torch_utils import torch_distributed_zero_first

from utils.augmentation import build_augmentation_pipeline

from typing import Union, List, Tuple, Dict, Any

# global parameters
IMAGE_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VIDEO_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files: List[Union[str, Path]]) -> float:
    """Returns a single hash value of a list of files"""
    # ensure pathlib object
    files = [Path(fl) for fl in files]
    # sum file statistics
    return sum([fl.stat().st_mtime_ns + fl.stat().st_size for fl in files if fl.is_file()])


def exif_size(img: Image. Image) -> Tuple[int, int]:
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(
        path,
        imgsz,
        batch_size,
        stride,
        opt,
        hyp=None,
        augment: bool = False,
        cache: bool = False,
        pad: float = 0.0,
        rect: bool = False,
        rank: int = -1,
        world_size: int = 1,
        workers: int = 8,
        image_weights: bool = False,
        quad: bool = False,
        prefix: str = '',
        augmentation_config: str = None,
        global_augmentation_probability: float = None,
        mosaic_augmentation: bool = True,
        n_keypoints: int = None,
):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):

        kwargs = {
            "path_to_data_info": path,
            "img_size": imgsz,
            "batch_size": batch_size,
            "augment": augment,
            "hyp": hyp,  # augmentation hyperparameters
            "rect": rect,  # rectangular training
            "cache_images": cache,
            "single_cls": opt.single_cls,
            "stride": int(stride),
            "pad": pad,
            "image_weights": image_weights,
            "prefix": prefix,
            "augmentation_config": augmentation_config,
            "augmentation_probability": global_augmentation_probability,
            "mosaic_augmentation": mosaic_augmentation,
            "n_keypoints": n_keypoints
        }

        dataset = LoadImagesAndLabels(**kwargs)

    batch_size = min(batch_size, len(dataset))
    n_workers = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn
    ) # FIXME: LoadImagesAndLabelsWithMasks.collate_fn4

    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMAGE_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VIDEO_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMAGE_FORMATS}\nvideos: {VIDEO_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            logging.debug(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        logging.debug(f"LoadImages: letterbox. img0.shape={img0.shape}, img.shape={img.shape}")
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        logging.debug(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, src in enumerate(sources):
            # Start the thread to read frames from the video stream
            logging.debug(f'{i + 1}/{n}: {src}... ', end='')
            url = eval(src) if src.isnumeric() else src
            if 'youtube.com/' in str(url) or 'youtu.be/' in str(url):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype="mp4").url
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {src}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            logging.info(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            logging.warning('Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths) -> List[Path]:
    # Define label paths as a function of image paths
    return [Path(x).with_suffix('.txt') for x in img_paths]


def read_annotations(info_file: Path, image_folder: Path = None) -> Tuple[List[Path], List[Path]]:
    images = []
    labels = []
    try:
        files = []
        if info_file.is_dir():  # dir
            files += list(info_file.glob("**/*.*"))
        elif info_file.is_file():  # file
            with open(info_file, "r", encoding="utf-8") as fid:
                lines = fid.read().strip().splitlines()
                # convert paths to pathlib objects
                lines = [Path(el) for el in lines]
                # add parent of the data info file if the path to the label file is not an absolute path.
                # (this is just a precaution)
                files += [el if el.is_absolute() else (info_file.parent / el).resolve() for el in lines]
        else:
            raise Exception(f'{info_file} does not exist')

        # files
        if image_folder is not None:
            # construct path
            image_folder = info_file.parent / image_folder
            if not image_folder.is_dir():
                raise Exception(f'{image_folder.as_posix()} is not a directory.')
            # assume all files to be annotations
            for fl in files:
                imgs = list(Path(image_folder).glob(f"{fl.stem}.*"))
                if len(imgs) > 0:
                    img = imgs[0].resolve()
                    if img.suffix.strip(".") in IMAGE_FORMATS:
                        images.append(img)
                        labels.append(fl)
        else:
            # assume all files to be annotations
            for fl in files:
                img = fl
                lbl = fl.with_suffix(".txt")
                if img.suffix.strip(".") in IMAGE_FORMATS:
                    images.append(img)
                    labels.append(lbl)
        # add files if they match one of the allowed image formats
        img_files = images
        assert img_files, f"No images found"
    except Exception as e:
        raise Exception(f'Error loading data from {info_file.as_posix()}: {e}')
    return images, labels


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
            self,
            path_to_data_info: Union[str, Path],
            img_size: Union[int, Tuple[int, int]] = 640,
            batch_size: int = 16,
            augment: bool = False,
            hyp=None,
            rect: bool = False,
            image_weights: bool = False,
            cache_images: bool = False,
            single_cls: bool = False,
            stride: int = 32,
            pad: float = 0.0,
            prefix: str = "",
            augmentation_config: Union[str, Path] = None,
            augmentation_probability: float = None,
            mosaic_augmentation: bool = True,
            n_keypoints: int = None,
            annotation_type: str = "bbox",
            shuffle: bool = False,
            path_to_images: Union[str, Path] = None
            ):
        self.img_size = [img_size, img_size] if isinstance(img_size, (int, float)) else img_size
        self.augment = augment
        self.augmentation_probability = augmentation_probability
        self.n_kpt = n_keypoints if n_keypoints else 0
        self.sz_label = 5 + 3 * self.n_kpt  # (cls, x, y, w, h) for bbox + [(x, y, visibility), ...] for keypoints
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect and mosaic_augmentation  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = -np.array(self.img_size)[::-1] // 2
        self.stride = stride
        self.path_data_info = Path(path_to_data_info)
        self.shuffle = shuffle
        self.path_to_images = Path(path_to_images) if path_to_images is not None else None

        # set augmentation functions by the python package albumentations
        if self.augment and augmentation_config:
            self.albumentations = Albumentations(
                config_file=augmentation_config,
                p=augmentation_probability,
                annotation_type=annotation_type
            )
        else:
            self.albumentations = None

        self.img_files, self.label_files = read_annotations(self.path_data_info, self.path_to_images)

        # Check cache
        if self.path_data_info.is_file():
            cache_path = self.path_data_info
        else:
            cache_path = Path(self.label_files[0]).parent
        cache_path = cache_path.with_suffix('.cache')

        cache_exists = cache_path.is_file() and cache_images
        if cache_exists:
            cache = torch.load(cache_path)  # load
            # check if cache was changed
            hash_of_files = get_hash(self.label_files + self.img_files)
            if cache['hash'] != hash_of_files:
                # files were changed
                cache_exists = False

        if not cache_exists:
            cache = self.cache_labels(cache_path, prefix)  # cache

        # Display cache
        n_found, n_missing, n_empty, n_corrupted, n_files = cache.pop('results')  # found, missing, empty, corrupted, total
        if cache_exists:
            msg = f"Scanning '{cache_path}' images and labels (hash: {cache['hash']}) ... " \
                  f"{n_found} found, {n_missing} missing, {n_empty} empty, {n_corrupted} corrupted"
            tqdm(None, desc=prefix + msg, total=n_files, initial=n_files)  # display cache results

        assert n_found > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels.'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n_images = len(shapes)  # number of images
        bi = np.floor(np.arange(n_images) / batch_size).astype(np.int32)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n_images
        self.indices = list(range(n_images))

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            self.segments = [self.segments[i] for i in irect]
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int32) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n_images
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n_images, [None] * n_images
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n_images)))
            pbar = tqdm(enumerate(results), total=n_images)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

        ## DEBUGGING
        assert len(self.segments) == len(self.labels)
        assert all([len(seg) == lbl.shape[0] for seg, lbl in zip(self.segments, self.labels) if seg])

    def cache_labels(self, path_to_cache: Union[str, Path] = './labels.cache', prefix: str = '') -> Dict[str, Any]:
        # ensure pathlib object
        path_to_cache = Path(path_to_cache)
        # Cache dataset labels, check images and read shapes
        cache = {}  # dict
        n_missing, n_found, n_empty, n_corrupted = 0, 0, 0, 0  # number missing, found, empty, duplicate / NaN

        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        n_files = 0
        for im_file, lb_file in pbar:
            n_files += 1
            # ensure pathlib object
            lb_file = Path(lb_file)

            try:
                # verify images
                img = Image.open(im_file)
                img.verify()  # PIL verify
                shape = exif_size(img)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert img.format.lower() in IMAGE_FORMATS, f'invalid image format {img.format}'

                # verify labels
                if lb_file.is_file():
                    n_found += 1  # label found

                    # read file with labels
                    with open(lb_file, "r") as fid:
                        lines_raw = fid.readlines()
                    lines = [ln.strip().split(" ") for ln in lines_raw]

                    # determine label typ from annotations
                    lengths = [len(el) for el in lines]
                    if all([el == lengths[0] for el in lengths]):
                        if lengths[0] == 5:  # class, x, y, w, h
                            label_type = "bbox"
                        elif (lengths[0] - 1) % 3 == 0 and self.n_kpt:  # TODO: keypoint flag is only required if all polygons are triangles...
                            label_type = "keypoint"
                        else:
                            label_type = "segment"
                    else:
                        label_type = "segment"

                    if label_type == "segment":  # is segment
                        # restructure if label is a "segment"
                        classes = np.array([x[0] for x in lines], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lines]  # (cls, xy1...)
                        # build lines as were expected in the first place
                        lines = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    else:
                        # cast to floats (classes are supposed to be integers though)
                        lines = np.array(lines, dtype=np.float32)

                    # verify labels
                    if len(lines):
                        if label_type == "keypoint":
                            msg = f'labels require 5 + 3 * #keypoints columns for keypoints ' \
                                  f'(here: {self.sz_label} for {self.n_kpt} keypoints)'
                        elif label_type == "segment":
                            msg = f'labels require 1 + 3 * #points columns for polygon segments (restructured to 1 + 4)'
                        else:  # bbox
                            msg = f'labels require 1 + 4 columns for bounding-boxes'
                        assert lines.shape[1] == self.sz_label, msg

                        assert (lines >= 0).all(), 'negative labels'
                        # bounding-boxes expect 4 (normalized) coordinates, keypoints 2 for each point
                        assert (lines[:, self.idx_c] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(lines, axis=0).shape[0] == lines.shape[0], 'duplicate labels'
                    else:
                        n_empty += 1  # label empty
                        lines = np.zeros((0, self.sz_label), dtype=np.float32)
                else:
                    # if path is not a file (to labels)
                    n_missing += 1  # label missing
                    lines = np.zeros((0, self.sz_label), dtype=np.float32)

                cache[im_file] = [lines, shape, segments]
            except Exception as e:
                n_corrupted += 1
                logging.warning(f'Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path_to_cache.parent / path_to_cache.stem}' images and labels... " \
                        f"{n_found} found, {n_missing} missing, {n_empty} empty, {n_corrupted} corrupted"
        pbar.close()

        if n_found == 0:
            logging.warning(f'No labels found in {path_to_cache}.')

        cache['hash'] = get_hash(self.label_files + self.img_files)
        cache['results'] = n_found, n_missing, n_empty, n_corrupted, n_files
        cache['version'] = 0.1  # cache version
        torch.save(cache, path_to_cache)  # save for next time
        logging.info(f"{prefix}New cache created: {path_to_cache} (hash: {cache['hash']})")
        return cache

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            random.shuffle(self.indices)
        index_ = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        masks = None
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = load_mosaic(self, index_)
            else:
                img, labels = load_mosaic9(self, index_)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index_)

            # Letterbox
            shape = self.batch_shapes[self.batch[index_]] if self.rect else self.img_size[::-1]  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index_].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels = self.transform_to_absolute_coordinates(labels, ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # Augment image space
            if not mosaic:
                img, labels = random_perspective(
                    img, labels,
                    degrees=hyp['degrees'],
                    translate=hyp['translate'],
                    scale=hyp['scale'],
                    shear=hyp['shear'],
                    perspective=hyp['perspective']
                )
            if self.albumentations is not None:
                img, labels, masks = self.albumentations(img, labels, masks)

            # Augment colorspace
            if (random.random() < self.augmentation_probability) and all([el in hyp for el in ["hsv_h", "hsv_s", "hsv_v"]]):
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

            if random.random() < hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], []
                while len(sample_labels) < 30:
                    sample_labels_, sample_images_, sample_masks_ = load_samples(
                        self,
                        random.randint(0, len(self.labels) - 1)
                    )
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_

                    if len(sample_labels) == 0:
                        break
                labels = pastein(img, labels, sample_labels, sample_images, sample_masks)

        n_labels = len(labels)  # number of labels
        if n_labels:
            labels = self.transform_to_relative_coordinates(labels, w=img.shape[0], h=img.shape[1])

        # if self.augment:  # ALBUMENTATIONS
        #     # flip up-down
        #     if random.random() < hyp['flipud']:
        #         img = np.flipud(img)
        #         if n_labels:
        #             labels[:, 2] = 1 - labels[:, 2]
        #
        #     # flip left-right
        #     if random.random() < hyp['fliplr']:
        #         img = np.fliplr(img)
        #         if n_labels:
        #             labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((n_labels, self.sz_label + 1))  # +1 to add batch number (as prefix)
        if n_labels:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        path = self.img_files[index_]

        return torch.from_numpy(img), labels_out, path, shapes, masks

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks = zip(*batch)  # transposed
        for i, lbl in enumerate(label):
            lbl[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, masks

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes, masks = zip(*batch)  # transposed  # FIXME: masks4
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                lbl = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lbl = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(lbl)

        for i, lbl in enumerate(label4):
            lbl[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4, masks

    @property
    def idx_c(self) -> List[int]:
        """index of coordinates in the label tensor"""
        return self.idx_c_bbx + self.idx_c_kpt

    @property
    def idx_c_bbx(self) -> List[int]:
        """index of coordinates of the bounding box (x, y, w, h) in the label tensor"""
        return list(range(1, 5))

    @property
    def idx_c_kpt(self) -> List[int]:
        """index of coordinates of the keypoints (x, y, visibility, ...) in the label tensor"""
        idx0_kpt = max(self.idx_c_bbx) + 1
        return [idx0_kpt + x for i in range(self.n_kpt) for x in (3 * i, 3 * i + 1)]

    def transform_to_absolute_coordinates(self, labels, w, h, padw, padh, segments=None):
        """make labels to absolute corner coordinates (lower-left + upper-right corner)"""
        sz_lbl = labels[0].size

        # bounding boxes
        # normalized xywh to pixel xyxy format
        labels[:, self.idx_c_bbx] = xywhn2xyxy(labels[:, self.idx_c_bbx], w, h, padw, padh)
        if segments is not None:
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        if sz_lbl == 5:
            # bounding box regression
            pass
        elif sz_lbl > 5 and (sz_lbl - 5) % 3 == 0:
            # keypoints
            n_kpt = (sz_lbl - 5) // 3
            assert self.n_kpt == n_kpt
            # FIXME: do we need to scale keypoints to absolute coordinates?
            # TODO: absolute scale relative to bounding box!
            xy1 = labels[:, [1, 2]]
            xy2 = labels[:, [3, 4]]
            bbx_size = xy2 - xy1
            bbx_size_repeated = np.hstack((bbx_size, ) * self.n_kpt)
            offset = np.hstack((xy1, ) * self.n_kpt)
            # labels[:, self.idx_c_kpt] = labels[:, self.idx_c_kpt] * bbx_size_repeated + offset
        elif sz_lbl > 5 and (sz_lbl - 5) % 2 == 0:
            # is polygon
            pass
        else:
            msg = "Unexpected size of labels. "
            if self.n_kpt:
                msg += f"Was expecting 5 + 3 * #keypoints for keypoints but was {sz_lbl}."
            else:
                msg += f"Was expecting length 5 for bounding boxes but was {sz_lbl}."
            raise ValueError(msg)
        return labels

    def transform_to_relative_coordinates(self, labels, w: int, h: int):
        """make labels to relative center coordinates (center point + width + height)"""
        sz_lbl = labels[0].size

        # bounding boxes
        # pixel xyxy format to normalized xywh
        labels[:, self.idx_c_bbx] = xyxy2xywh(labels[:, self.idx_c_bbx])  # convert xyxy to xywh
        labels[:, [2, 4]] /= w  # normalized height 0-1
        labels[:, [1, 3]] /= h  # normalized width 0-1

        if sz_lbl == 5:
            # bounding boxes
            pass
        elif sz_lbl > 5 and (sz_lbl - 5) % 3 == 0:
            # keypoints
            n_kpt = (sz_lbl - 5) // 3
            assert self.n_kpt == n_kpt
            # TODO: relative to bounding box!
            # img_size = [w, h] * self.n_kpt
            # labels[:, self.idx_c_kpt] = labels[:, self.idx_c_kpt] / img_size
        else:
            msg = "Unexpected size of labels. "
            if self.n_kpt:
                msg += f"Was expecting 3 * #keypoints for keypoints but was {sz_lbl}."
            else:
                msg += f"Was expecting length 5 for bounding boxes but was {sz_lbl}."
            raise ValueError(msg)
        return labels


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        p2img = Path(self.img_files[index])
        img = cv2.imread(p2img.as_posix())  # BGR
        assert img is not None, f"Image Not Found {p2img.as_posix()}"

        h0, w0 = img.shape[:2]  # orig hw
        resize_factor = np.array(self.img_size) / (w0, h0)  # resize image to img_size
        if any(resize_factor != 1):  # always resize down, only resize up if training with augmentation
            # interpolation method (code)
            interp = cv2.INTER_AREA if (any(resize_factor < 1) and not self.augment) else cv2.INTER_LINEAR
            # resize factor
            resize_factor_min = min(resize_factor)
            img_size_new = np.array(img.shape[:2]) * resize_factor_min
            # interpolate
            img = cv2.resize(img, img_size_new[::-1].astype(int), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    img_w, img_h = self.img_size
    # pick random center to crop the image later
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x, s in zip(self.mosaic_border, (img_h, img_w))]  # mosaic center x, y

    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((img_h * 2, img_w * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_w * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_h * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_w * 2), min(img_h * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # make absolute coordinates (lower-left + upper-right corner)
            labels = self.transform_to_absolute_coordinates(labels, w, h, padw, padh, segments)
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * [2 * img_h, 2* img_w], out=x)  # clip when using random_perspective()
        # FIXME: check with segments
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    # sample_segments(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(
        img4, labels4, segments4,
        degrees=self.hyp['degrees'],
        translate=self.hyp['translate'],
        scale=self.hyp['scale'],
        shear=self.hyp['shear'],
        perspective=self.hyp['perspective'],
        border=self.mosaic_border
    )  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    img_w, img_h = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices

    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((img_h * 3, img_w * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = (img_w, img_h, img_w + w, img_h + h)  # xmin, ymin, xmax, ymax (base) coordinates
            # c = (img_w, img_h, 2 * img_w, 2 * img_h)  # ideal
        elif i == 1:  # top
            # c = s, s - h, s + w, s  # original
            c = (img_w, img_h - h, img_w + w, img_h)
            # c = (img_w, 0, 2 * img_w, img_h)  # ideal
        elif i == 2:  # top right
            c = (img_w + wp, img_h - h, img_w + wp + w, img_h)
            # c = s + wp, s - h, s + wp + w, s  # original
            # c = (2 * img_w, 0, 3 * img_w, img_h)  # ideal
        elif i == 3:  # right
            c = (img_w + w0, img_h, img_w + w0 + w, img_h + h)
            # c = s + w0, s, s + w0 + w, s + h  # original
            # c = (2 * img_w, img_h, 3 * img_w, 2 * img_h)  # ideal
        elif i == 4:  # bottom right
            c = (img_w + w0, img_h + hp, img_w + w0 + w, img_h + hp + h)
            # c = s + w0, s + hp, s + w0 + w, s + hp + h  # original
            # c = (2 * img_w, 2 * img_h, 3 * img_w, 3 * img_h)  # ideal
        elif i == 5:  # bottom
            c = (img_w + w0 - w, img_h + h0, img_w + w0, img_h + h0 + h)
            # c = s + w0 - w, s + h0, s + w0, s + h0 + h  # original
            # c = (img_w, 2 * img_h, 2 * img_w, 3 * img_h)  # ideal
        elif i == 6:  # bottom left
            c = (img_w + w0 - wp - w, img_h + h0, img_w + w0 - wp, img_h + h0 + h)
            # c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h  # original
            # c = (0, 2 * img_h, img_w, 3 * img_h)  # ideal
        elif i == 7:  # left
            c = (img_w - w, img_h + h0 - h, img_w, img_h + h0)  # original
            # c = s - w, s + h0 - h, s, s + h0
            # c = (0, img_h, img_w, 2 * img_h)  # ideal
        elif i == 8:  # top left
            c = (img_w - w, img_h + h0 - hp - h, img_w, img_h + h0 - hp)
            # c = s - w, s + h0 - hp - h, s, s + h0 - hp  # original
            # c = (0, 0,  img_w, img_h)  # ideal

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, x)) for x in [img_h, img_w]]  # mosaic center x, y
    # crop around random center
    img9 = img9[yc:(yc + 2 * img_h), xc:(xc + 2 * img_w)]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    cxy = np.array([xc, yc])  # centers
    segments9 = [x - cxy for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * [2 * img_h, 2 * img_w], out=x)  # clip when using random_perspective()
        # FIXME: check with segments
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    # img9, labels9, segments9 = remove_background(img9, labels9, segments9)
    img9, labels9, segments9 = copy_paste(img9, labels9, segments9, probability=self.hyp['copy_paste'])
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def load_samples(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    sample_labels, sample_images, sample_masks = sample_segments(img4, labels4, segments4, probability=0.5)

    return sample_labels, sample_images, sample_masks


def copy_paste(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img, labels, segments


def remove_background(img, labels, segments):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    h, w, c = img.shape  # height, width, channels
    im_new = np.zeros(img.shape, np.uint8)
    img_new = np.ones(img.shape, np.uint8) * 114
    for j in range(n):
        cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)

        i = result > 0  # pixels to replace
        img_new[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img_new, labels, segments


def sample_segments(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0, w - 1), l[2].astype(int).clip(0, h - 1), l[3].astype(int).clip(0, w - 1), l[
                4].astype(int).clip(0, h - 1)

            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue

            sample_labels.append(l[0])

            mask = np.zeros(img.shape, np.uint8)

            cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3], box[0]:box[2], :])

            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

            sample_images.append(mask[box[1]:box[3], box[0]:box[2], :])

    return sample_labels, sample_images, sample_masks


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scaleFill: bool = False,
        scaleup: bool = True,
        stride: int = 32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(
        img,
        targets=(),
        segments=(),
        degrees=10,
        translate=.1,
        scale=.1,
        shear=10,
        perspective=0.0,
        border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def pastein(image, labels, sample_labels, sample_images, sample_masks):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
        else:
            ioa = np.zeros(1)

        if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin + 20) and (
                ymax > ymin + 20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_labels) - 1)
            # print(len(sample_labels))
            # print(sel_ind)
            # print((xmax-xmin, ymax-ymin))
            # print(image[ymin:ymax, xmin:xmax].shape)
            # print([[sample_labels[sel_ind], *box]])
            # print(labels.shape)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
            r_w = int(ws * r_scale)
            r_h = int(hs * r_scale)

            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin + r_h, xmin:xmin + r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int32).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    # print(sample_labels[sel_ind])
                    # print(sample_images[sel_ind].shape)
                    # print(temp_crop.shape)
                    box = np.array([xmin, ymin, xmin + r_w, ymin + r_h], dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])

                    image[ymin:ymin + r_h, xmin:xmin + r_w] = temp_crop

    return labels


class Albumentations:
    # (optional, only used if package is installed)
    def __init__(
            self,
            config_file: Union[str, Path],
            p: float = None,
            annotation_type: str = "bbox",
            verbose: bool = True
    ):
        self.transform = None
        import albumentations as A

        trafo_fncs = build_augmentation_pipeline(config_file, probability=p, verbose=verbose)

        # TODO: augment bounding boxes + image with albumentations and transform corresponding keypoints manually afterwards
        if annotation_type in ["bbox", "segment"]:
            # yolo: normalized [x0, y0, w, h]
            # pascal_voc: [x_min, y_min, x_max, y_max]
            args = {"bbox_params": A.BboxParams(format="pascal_voc", label_fields=["class_labels"])} # this is xyxy coordinates
        elif annotation_type == "keypoint":
            args = {"keypoint_params": A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False)}
        else:
            raise ValueError(f"Unknown annotation format: {annotation_type}. "
                             f"Valid formats are 'bbox' for bounding boxe, 'keypoint' for keypoints, or"
                             f"'segment' for polygons for segmentation.")
        self.transform = A.Compose(trafo_fncs, **args)
        self.annotation_type = annotation_type

        logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))

    def __call__(
            self,
            imgs: np. ndarray,
            labels: np. ndarray,
            masks: np.ndarray = None,
            p: float = 1.0
    ):
        if self.transform and random.random() < p:
            new = self.transform(
                image=imgs,
                bboxes=labels[:, 1:5],
                class_labels=labels[:, 0],
                masks=[el.astype(np.uint8) for el in masks] if isinstance(masks, np.ndarray) and len(masks.shape) > 2 else masks
            )  # transformed
            # TODO: add keypoints
            imgs = new["image"]
            labels = np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
            if isinstance(new["masks"], list):
                if len(new["masks"]) > 1:
                    masks = np.stack(new["masks"])
                else:
                    masks = np.array(new["masks"])
            else:
                masks = new["masks"]
        return imgs, labels, masks


def create_folder(path: Union[str, Path] = './new'):
    # Create folder
    if Path(path).exists():
        shutil.rmtree(path)  # delete output folder
    Path(path).mkdir()  # make new output folder


def flatten_recursive(path='../coco'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMAGE_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int32)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco')
    Arguments
        path:           Path to images directory
        weights:        Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in IMAGE_FORMATS], [])  # image files only
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing

    logging.debug(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file


def load_segmentations(self, index):
    key = '/work/handsomejw66/coco17/' + self.img_files[index]
    # /work/handsomejw66/coco17/
    return self.segs[key]
