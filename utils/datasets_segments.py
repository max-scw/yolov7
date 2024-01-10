import os
import logging
import random
import numpy as np
import cv2

import torch

from utils.general import (
    xywhn2xyxy,
    xyn2xy,
    # colorstr
    clip_coords
)

from utils.torch_utils import torch_distributed_zero_first
from utils.datasets import (
    augment_hsv,
    copy_paste,
    InfiniteDataLoader,
    letterbox,
    load_image,
    LoadImagesAndLabels,
)
from utils.segment import augmentations as segaug   # TODO: check code from mask branch


def create_dataloader(
        path,
        imgsz,
        batch_size,
        stride,
        opt,
        hyp: dict = None,
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
        predict_masks: bool = False,
        downsample_ratio=1,
        overlap: bool = False
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

        if predict_masks:
            kwargs["annotation_type"] = "segment"
            kwargs["downsample_ratio"] = downsample_ratio
            kwargs["overlap"] = overlap
            dataset = LoadImagesAndLabelsWithMasks(**kwargs)
        else:
            kwargs["annotation_type"] = "keypoint" if n_keypoints > 0 else "bbox"
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
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabelsWithMasks.collate_fn
    )  # TODO: LoadImagesAndLabelsWithMasks.collate_fn4

    return dataloader, dataset


class LoadImagesAndLabelsWithMasks(LoadImagesAndLabels):  # for training/testing
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**{ky: val for ky, val in kwargs.items() if ky not in ["overlap", "downsample_ratio"]})

        self.downsample_ratio = kwargs["downsample_ratio"] if "downsample_ratio" in kwargs else 1
        self.overlap = kwargs["overlap"] if "overlap" in kwargs else False

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        masks = None
        is_rel_coords = True
        if mosaic:
            # Load mosaic
            img, labels, segments = self.load_mosaic(index)
            shapes = None
            # TODO: integrate masks

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels, segments = segaug.mixup(img, labels, segments, *self.load_mosaic(random.randint(0, self.n - 1)))
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)
            # image, original (h, w), resized (h, w)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            if len(segments):
                # normalized segment points to absolute pixels
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                is_rel_coords = False

        if self.augment:
            img, labels, segments = segaug.random_perspective(
                img,
                targets=labels,
                segments=segments,
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
                perspective=hyp["perspective"],
            )

        n_labels = len(labels)  # number of labels
        if n_labels:
            # labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(
                    img.shape[:2],
                    segments,
                    downsample_ratio=self.downsample_ratio
                )
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(
                    img.shape[:2],
                    segments,
                    color=1,
                    downsample_ratio=self.downsample_ratio
                )

        if masks is not None and len(masks) > 0:
            # albumentations requires numpy.ndarray anyway
            pass
        else:
            masks = np.zeros(
                shape=(
                    1 if self.overlap else n_labels,
                    img.shape[0] // self.downsample_ratio,
                    img.shape[1] // self.downsample_ratio
                ), dtype=np.uint8
            )
            # print(f"DEBUG: no masks. Create black mask: masks.shape={masks.shape}, labels.shape={labels.shape}")

        if self.augment:
            if self.albumentations is not None:
                # Albumentations
                # there are some augmentation that won't change boxes and masks,
                # so just be it for now.
                img, labels, masks = self.albumentations(img, labels, masks)
                n_labels = len(labels)  # update after albumentations

            # to torch.Tensor
            masks = torch.from_numpy(masks)

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # TODO: add paste_in augmentation

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if n_labels:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if n_labels:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        if not is_rel_coords and n_labels:
            labels = self.transform_to_relative_coordinates(labels, w=img.shape[0], h=img.shape[1])
            is_rel_coords = True

        labels_out = torch.zeros((n_labels, 6))
        if n_labels:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        path = self.img_files[index]

        return torch.from_numpy(img), labels_out, path, shapes, masks

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
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
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4, segments4 = segaug.random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border  # border to remove
        )
        return img4, labels4, segments4

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks = zip(*batch)  # transposed
        if isinstance(masks, tuple) and all([isinstance(el, torch.Tensor) for el in masks]):
            batched_masks = torch.cat(masks, 0)
        elif isinstance(masks, tuple) and all([isinstance(el, np.ndarray) for el in masks]):
            batched_masks = torch.from_numpy(np.concatenate(masks))
        else:
            batched_masks = masks

        for i, lbl in enumerate(label):
            lbl[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks


# ----- ONLY FOR MASKS
def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
            dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def xyxy2xywhn(x: np.ndarray, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    # FIXME: only for masks!
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y