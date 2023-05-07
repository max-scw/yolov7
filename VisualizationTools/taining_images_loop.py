from pathlib import Path
import os
import argparse
import yaml

from utils.datasets import create_dataloader
from utils.plots import plot_images
from utils.general import check_img_size

from utils.debugging import print_debug_msg


def image_loop(train_path: Path, opt):
    gs = 32  # grid size
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    dataloader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt,
                                            hyp=hyp,
                                            cache=opt.cache_images,
                                            rect=opt.rect,
                                            rank=-1,
                                            world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1,
                                            workers=opt.workers,
                                            image_weights=opt.image_weights,
                                            quad=opt.quad,
                                            prefix='train: ',
                                            augment=not opt.no_augmentation,
                                            yolov5_augmentation=True if opt.albumentations_probability == 0.01 else False,
                                            albumentation_augmentation_p=opt.albumentations_probability,
                                            mosaic_augmentation=not opt.no_mosaic_augmentation
                                            )

    for epoch in range(opt.start_epoch, opt.epochs):  # epoch
        # dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        for i, (imgs, targets, paths, _) in pbar:  # batch
            path_to_export = Path(opt.export_training_images) / opt.name
            if not path_to_export.is_dir():
                path_to_export.mkdir()
            p2fl = path_to_export / f"{opt.name}_e{epoch}_b{i}.jpg"
            print_debug_msg(f"{p2fl.as_posix()}: {imgs.shape}")
            plot_images(imgs, targets, fname=p2fl.as_posix(), max_subplots=opt.batch_size, aspect_ratio=16 / 9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--no-augmentation', action='store_true', help='Do not augment training data')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument("--albumentations_probability", type=float, default=0.01,
                        help="Probability to apply data augmentation based on the albumentations package.")
    parser.add_argument("--export-training-images", type=str, default="VisualizationTools",
                        help="Folder where to export the (augmented) training images to.")
    parser.add_argument("--no-mosaic-augmentation", action='store_true',
                        help="Do not apply mosaic augmentation.")

    opt = parser.parse_args()

    # opt.hyp = "data/hyp.scratch.custom.yaml"
    # opt.weights = "trained_models/yolov7-tiny.pt"
    # opt.albumentations_probability = 0.5
    # opt.batch_size = 66  # 62, 41
    # opt.epochs = 5000
    # opt.start_epoch = 1001
    # opt.name = "TEST_tiny-yolo7"
    # opt.export_training_images = ""

    image_loop(Path("Trn.txt"), opt)
    # NOTE: Train/Test/Validation files must be moved to this folder and specify the path relative to here

