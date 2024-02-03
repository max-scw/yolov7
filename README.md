# Slim YOLOv7

Fork of the original [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) by [Kin-Yiu Wong](https://github.com/WongKinYiu). 

Supposed to present a concise version of the original code for greater usability. 
All credit must go to the original authors. This version was created out of the endeavor to understand the code in detail. It is due to my limited comprehension that part of the code was re-formatted, re-written, re-named or deleted to align with my understanding of the academic paper ([YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696), [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)), Python and Neural Networks.

Consider this project rather a "for dummies" fork.

## Quick start
[train.py](train.py) trains a model and saves the weights (and checkpoints) to *./runs/train/<NAME>/weights* (<NAME> can be set by the `--name` argument.)

The network structure is stored in the [cfg](./cfg/) folder. Relevant are [YOLOv7](./cfg/training/yolov7.yaml) or [tiny YOLOv7](./cfg/training/yolov7.yaml).
Hand the desired model structure to [train.py](train.py) by the `--cfg ./cfg/training/yolov7-tiny.yaml` argument. 

Provide [hyperparameters](####Hyperparameters) with `--hyp ./data/hyp.tiny.yaml` and the configuration of the [albumentations](https://albumentations.ai/)-based [augmentation](####Augmentation) with `--hyp ./data/augmentation-yolov7.yaml`. 

[Data](##Data) is stored in a file like [myYOLO.yaml](data%2FmyYOLO.yaml) or [coco.yaml](data%2Fcoco.yaml) that stores the number of classes, their names and points to the dataset files for training, validation, and testing which eventually point to the images.
Use `--data data/myYOLO.yaml` to hand it over.

Use a pretrained weights with `--weights yolov7.pt`, e.g. for [YOLOv7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) or [YOLOv7-tiny](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt) pretrained on [MS COCO](https://cocodataset.org/).

See the respective sections for details in [Training](###Training).




### Idea (quick pass-through the original paper [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696))
*Core idea*: "bag of freebies", i.e. invest at training time to reduce inference time => reduce about 40% parameters and 50% computation compared to state-of-the-art real-time object detector.
To explore the "bag of freebies" (i.e. the hyperparameters), an  evolution-inspired search is started (well, a grid search would indeed be exhaustive.)

### project structure
 ````
 YOLOv7
+-- cfg  # model architectures
   +-- baseline  # configuration of reference models
   +-- deploy  # the same as +-- training
   +-- training  # holds the model configurations
       |-- yolo7.yaml  # model configuration
       |-- yolo7-tiny.yaml  # model configuration
+-- data  # configurations for individual use cases
   |-- augmentation-yolov7.yaml  # augmentation configuration
   |-- coco.yaml  # dataset configuration for MS COCO dataset
   |-- hyp.scratch.custom.yaml  # default hyperparameters to train YOLOv7 from scratch
   |-- hyp.tiny.yaml  # default hyperparameters for tiny YOLOv7
   |-- myYOLO.yaml  # exemplary dataset configuration for custom dataset
+-- (dataset)  # you may want to hold all data here
   +-- (MyDataSet)  # use case 1
      +-- (data)
         |-- (Image1.jpg)  # image
         |-- (Image1.txt)  # label
         |-- ...
      |-- (Trn.txt)  # lists path to images for training,
      |-- (Tst.txt)  # ... for testing,
      |-- (Val.txt)  # ... for validation. File names must match with data/myYOLO.yaml. train.py creates cache files here.
+-- docs  # holds images to illustrate README files
+-- models  # module to build the PyTorch model by parsing a YAML file above
+-- (runs)  # stores results of training and testing calls. Created automatically
   +-- (train)
      +-- (NAME)  # named folder per call
         +-- (weights)  # checkpoints and weights
            |-- best.pt  # best model weights (is checkpoint while training and reduced to weights only when finished)
+-- utils  # helper functions, e.g. for loss, augmentation or plotting
+-- VisualizationTools  # minimal code to create videos
|-- augment_validation_set.py  # applies data augmentation to validation set. Only use if dataset is very very small
|-- detect.py  # inference
|-- export.py  # export to ONNX for deployment
|-- hubconf.py  # PyTorch hub see original implementation
|-- requirements.txt  pip requirements
|-- test.py  # call for evaluation
|-- train.py  # trains the model
|-- watchdog_service.py
````


## Installation
Install the required dependencies (to your virtual environment, I hope this goes without saying ;))
````shell
pip install -r requirements.txt
````
BTW, the model relies on [PyTorch](https://pytorch.org/). Check out how to export the trained model to ONNX format with [export.py](./export.py) to integrate it to other NN-frameworks (or without). 

Tested on Python 3.9 - 3.11 with [PyTorch](https://pytorch.org/). 1.3,  2.1, and 2.2.
## Usage
### Training
You can train a network with [train.py](./train.py) -- so far so obvious.
Example call:
````shell
python train.py --device 'cpu' --batch-size 16 --data data/myYOLO.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name my-yolov7 --hyp data/hyp.scratch.custom.yaml --epochs 300 --adam
````
This also provides already trained weights [`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) on the [MS COCO](https://cocodataset.org/) dataset (**transfer learning**). Note that one should first train the head of the model if the head differs from the pre-trained dataset (e.g. different number of classes.) For this, freeze the body and train the network for some epochs `python train.py --freeze 60 ...` (check for how many layers the weights were transferred. This is printed on the console when setting up the network. For a *tiny YOLOv7* I have godf experiences with freezing the first 75 layers.)
Afterward, run the training a second time without freezing the layers and use the final weights of the previous run as input. This **fine-tuning** very much likely further improves accuracy. This two-stage training should require less than 1/10 of the time of training the model from only partly matching weights (doing transfer learning with a random head) and of course much less than training everything from scratch.


The model structure / architecture is provided as a YAML file: `--cfg ./cfg/training/yolov7-tiny.yaml`. It is parsed when setting up the training (by the [yolo.py](./models/yolo.py) in the [models](./models/) folder.) This provides great flexibility when evaluating different model structures, e.g. to benchmark against baseline models (e.g. [YOLOv3](./cfg/baseline/yolov3.yaml), [YOLOv4](./cfg/baseline/yolov4-csp.yaml), [YOLOE](./cfg/baseline/yolor-e6.yaml) of the same authors, thus with the same backend "[Darknet](https://pjreddie.com/darknet/)") or evaluate scaled structures ([YOLOv7-W6](./cfg/training/yolov7-w6.yaml), [YOLOv7-E6](./cfg/training/yolov7-e6.yaml), [YOLOv7-D6](./cfg/training/yolov7-d6.yaml).)

The data configuration is specified in another YAML file: `--data data/myYOLO.yaml`, where one can specify the number of classes, the class mapping and the files that point to the training / validation / test data. 

How the training data is augmented can be specified in a third file (`--augmentation-config data/augmentation-yolov7.yaml` holds the configuration for the original augmentation). Since use cases in industry usually don't have the luxury of millions of images, more augmentation (and with higher probability) is required. In this way, one can easily adjust the augmentation to their needs and what applies for a given use case.


All code can be accessed conveniently by command line commands and adjusted with extra arguments. Simply list all options with `python train.py --help`
```text
usage: train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--img-size IMG_SIZE [IMG_SIZE ...]] [--rect] [--resume [RESUME]] [--nosave] [--notest]   
                [--noautoanchor] [--evolve] [--evolve-generations EVOLVE_GENERATIONS] [--bucket BUCKET] [--cache-images] [--image-weights] [--device DEVICE] [--multi-scale] [--single-cls] [--adam] [--sync-bn]     
                [--local_rank LOCAL_RANK] [--workers WORKERS] [--project PROJECT] [--entity ENTITY] [--name NAME] [--exist-ok] [--quad] [--linear-lr] [--label-smoothing LABEL_SMOOTHING] [--save_period SAVE_PERIOD]
                [--save-best-n-checkpoints SAVE_BEST_N_CHECKPOINTS] [--freeze FREEZE [FREEZE ...]] [--no-augmentation] [--v5-metric] [--augmentation-probability AUGMENTATION_PROBABILITY]                           
                [--augmentation-config AUGMENTATION_CONFIG] [--export-training-images EXPORT_TRAINING_IMAGES] [--no-mosaic-augmentation] [--masks] [--save-not-every-epoch] [--process-title PROCESS_TITLE]          
                                                                                                                                                                                                                     
options:                                                                                                                                                                                                             
  -h, --help            show this help message and exit
  --weights WEIGHTS     initial weights path
  --cfg CFG             model.yaml path
  --data DATA           data.yaml path
  --hyp HYP             hyperparameters path
  --epochs EPOCHS
  --batch-size BATCH_SIZE
                        total batch size for all GPUs
  --img-size IMG_SIZE [IMG_SIZE ...]
                        [train, test] image sizes
  --rect                rectangular training
  --resume [RESUME]     resume most recent training
  --nosave              only save final checkpoint
  --notest              only test final epoch
  --noautoanchor        disable autoanchor check
  --evolve              evolve hyperparameters
  --evolve-generations EVOLVE_GENERATIONS
                        how many generations / iterations to evolve hyperparameters
  --bucket BUCKET       gsutil bucket to upload the file the the results of the evolutionary optimization to.
  --cache-images        cache images for faster training
  --image-weights       use weighted image selection for training
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --multi-scale         vary img-size +/- 50%
  --single-cls          train multi-class data as single-class
  --adam                use torch.optim.Adam() optimizer
  --sync-bn             use SyncBatchNorm, only available in DDP mode
  --local_rank LOCAL_RANK
                        DDP parameter, do not modify
  --workers WORKERS     maximum number of dataloader workers
  --project PROJECT     save to project/name
  --entity ENTITY       W&B entity
  --name NAME           save to project/name
  --exist-ok            existing project/name ok, do not increment
  --quad                quad dataloader
  --linear-lr           linear LR
  --label-smoothing LABEL_SMOOTHING
                        Label smoothing epsilon
  --save_period SAVE_PERIOD
                        Log model after every "save_period" epoch
  --save-best-n-checkpoints SAVE_BEST_N_CHECKPOINTS
                        How many checkpoints should be saved at most?
  --freeze FREEZE [FREEZE ...]
                        Freeze layers: backbone of yolov7=50, first3=0 1 2
  --no-augmentation     Do not augment training data
  --v5-metric           assume maximum recall as 1.0 in AP calculation
  --augmentation-probability AUGMENTATION_PROBABILITY
                        Probability to apply data augmentation based on the albumentations package.
  --augmentation-config AUGMENTATION_CONFIG
                        Path to config file with specifications for transformation functions to augment the trainig data.
  --export-training-images EXPORT_TRAINING_IMAGES
                        Folder where to export the (augmented) training images to.
  --no-mosaic-augmentation
                        Do not apply mosaic augmentation.
  --masks               Models polygons within the bounding boxes.
  --save-not-every-epoch
                        Does not save every epoch as last.pt
  --process-title PROCESS_TITLE
                        Names the process
```

![screencast_train.mp4](docs%2Fscreencast_train.mp4)

<video width="320" height="180" controls>
  <source src="docs/screencast_train.mp4" type="video/mp4">
 <p>Screencast of training start</p>
</video>

There are some extensions, e.g. for limiting the number of checkpoints to be stored (in order to avoid good old storage overflow) with `--save-best-n-checkpoints`. 
If you train on a dataset with only a few hundred examples, the number of epochs is usually much larger, thus cluttering the hard drive with checkpoints actually is a thread.

#### Hyperparameters
YOLOv7 conveniently offers a config file to specify the hyperparameters for training. Examples can be found in [data](data): [data/hyp.tiny.yaml](data%2Fhyp.tiny.yaml) or [data/hyp.scratch.tiny.yaml](data%2Fhyp.tiny.yaml).

There are two larger blocks of hyperparameters. Some address the loss functions: `box`, `cls`, `obj`, and `add` are the gains how to weight the loss of the bounding box position, its predicted class, the objectiveness and the additional points in the overall loss. The latter is newly added for segmentation and key-point detection (its default value is the same as `box`.) The parameters `cls_pw` and `obj_pw` are the gains for the binary cross entropy loss. Furthermore, if the slightly more complex over-the-air loss function should be used: `loss_ota` (0 / 1)
One can add a focal loss by setting the parameter `fl_gamma > 0.0` which is the exponential scaling factor (gamma). Per default the original focal loss is used. If one want to use the quality focal loss, please set `qfl: 1` (my personal experience is that it yields slightly better results but focal loss is only required when dealing with highly unbalanced classes.)
The other block of hyperparameters concerns augmentation. There is 
- an augmentation in HSV-space (hue, saturation, value): `hsv_h`, `hsv_s`, `hsv_v`,
- a perspective distortion: `degrees` (+/- deg), `translate` (+/- fraction), `scale` (+/- gain), `shear` (+/- deg), `perspective` (+/- fraction; range 0-0.001),
- probability of stacking multiple images together as a mosaic: `mosaic`,
- `mixup`
- probability of `copy_paste`
- probability of pasting elements of one image over another image slightly translucent: `paste_in`
- (default) probability for all transformations of the [albumentations](https://albumentations.ai/)-package (see below for details): `ablumentation_p`

The hyperparameters can be optimized with a genetic algorithm. This is the main idea of version 7 (the "bag of freebies"). Optimizing the hyperparameters increases the training effort tremendously but does not sacrifice inference effort.
The flag `--evolve` when calling `python train.py` triggers the evolutionary optimization. The number of generations can be adjusted by `--evolve-generations=300` (original implementation: 300).
However, default parameters are a good start for most cases. I recommend to focus on configuring the augmentation transformations as described below.

#### Augmentation
**The most interesting extension** though is the much more elaborate image augmentation using the [albumentations](https://albumentations.ai/)-package. It offers a neat bouquet of transformations from 
- artificial noise: imitating electric camera noise with [ISONoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ISONoise) or blur with [AdvancedBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.AdvancedBlur), simple [GaussNoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussNoise), or a specialized [Defocus](https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.blur.transforms.Defocus) transform,
- illumination changes like [RandomBrightnessContrast](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast), [RandomToneCurve](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomToneCurve), or local histogram equalization with [CLAHE](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CLAHE),
- geometric transforms ([HorizontalFlip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HorizontalFlip), [VerticalFlip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.VerticalFlip), or [ShiftScaleRotate](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate)),
- image deterioration, e.g. by compressing the image down and up with [ImageCompression](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ImageCompression).

There are many more transformations available. Since the permissible transformations are individual for every use case (e.g. not in every case the images should be flipped or rotated; some applications require a higher variation in illumination; etc.), transformations can be specified in a YAML-file for every project / dataset: [data/augmentation-yolov7.yaml](data%2Faugmentation-yolov7.yaml).
It starts with the library (currently only albumentations) and may use every available transformation. Parameters can be specified for every transformation; if not, default values of the library are used.
`````text
albumentations:
  ISONoise:  # camera sensor noise
    color_shift: (0.01, 0.05)
#    intensity: (0.1, 0.6)
`````
One can specify the probability of every transformation explicitly (`p: 0.5`) or set as global variable either in the hyperparameters (e.g. [data/hyp.tiny.yaml](data%2Fhyp.tiny.yaml)) or via `--augmentation-probability`. Take in mind that the probability applies to every transformation independently, i.e. the more transformations there are, the lower can (and should) be the transformation probability in oder to not completely distort the image.
The original YOLOv7 uses only few transformation from the albumentations package with only a very small probability (`--augmentation-probability=0.05`). But they used on the [MS COCO](https://cocodataset.org/) dataset with millions of images. If you have much fewer images (let's say only a couple of hundred), you should use more transformations and a little higher probability (I like 0.1-0.25 for the example config in [data/augmentation-yolov7.yaml](data%2Faugmentation-yolov7.yaml).)

(*Note*: the native vertical and horizontal flip in YOLOv7 was deactivated for consistency. Therese two hyperparameters no longer have an effect. The other augmentations newly introduced by YOLO such as *mosaic*, *mixup*, *paste-in*, and *perspective* are kept and apply on top of the albumentations transformation.)

### Inference
Once you have a trained model, inference is done in [detect.py](./detect.py):

On video:
``` shell
python detect.py --weights best.pt --conf 0.25 --img-size 640 --source <path to video>
```

On image:
``` shell
python detect.py --weights best.pt --conf 0.25 --img-size 640 --source <path to image>
```

I've added a little watchdog system that runs the model every time a new image is placed in the folder that is monitored (e.g. if you have a ftp-server where new images arrive). Check it out with
``` shell
python watchdog_service.py --weights best.pt --conf-thres 0.25 --img-size 640 --folder <path to folder>
```
The script is a simple infinite loop looking for newly created files in a folder and otherwise `sleep`s for a given time.
Use the `--class-colors` and `--interval` arguments to specify the colors of the bounding boxes and the time interval in which the watchdog should look for new files-

````text
usage: watchdog_service.py [-h] [--weights WEIGHTS] [--img-size IMG_SIZE] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES] [--folder FOLDER] [--class-colors CLASS_COLORS] [--interval INTERVAL]

options:
  -h, --help            show this help message and exit
  --weights WEIGHTS     Path to model weights
  --img-size IMG_SIZE   Size of displayed image (in pixels)
  --conf-thres CONF_THRES
                        Object confidence threshold
  --iou-thres IOU_THRES
                        IOU threshold for NMS
  --folder FOLDER       Path to folder that should be watched for any changes
  --class-colors CLASS_COLORS
                        Path to text file that lists the colors for each class.
  --interval INTERVAL   Time interval to check for new files in seconds.
````

### Deployment
The original work provides a nice integration for deploying the network primary to an [Nvidia Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) (an open-source project by [Nvidia on github](https://github.com/triton-inference-server/server) for a containerized server that distributes the workload of multiple models and requests on large Nvidia GPUs.) As CPUs perfectly suit my needs for inference, I barely had a look into this convenient deployment framework.  

There also exists the option to export the model as [Open Neural Network eXchange (ONNX)](https://onnx.ai/) format with [export.py](./export.py). I have experienced issued on the Non-Maximum Suppression algorithm, most likely due to my limited knowledge. However if there actually might be a patch needed, the original project is presumably the better address. Therefore, the deployment folder was not forked to this project.


## Data
The **data** is stored in a separate folder [data](./data/) that configures the paths to training, validation, and test data and core variables of the model (number and names of the classes.)
The YAML file for configuring the data (e.g. for [MS COCO](./data/coco.yaml)) points to separate text files for the training, validation, and test sets. Each file simply lists the paths to the images and their labels which are stored as text files with the same name as their corresponding image (placed in the same folder.) See the files [Trn.txt](./Trn.txt), [Val.txt](./Val.txt), and [Tst.txt](./Tst.txt) as an example.
A label file for a bounding box looks like this:
```text
12 0.5 0.5 0.1 0.05
2 0.3 0.8 0.1 0.1
```
All YOLO-models use center coordinates (x-y-w-h) relative to the image size. A bounding box is therefore specified by the coordinates of its center (x, y) and the width and height of the box (w, h).

Annotation: Every image has its own corresponding label file formatted as followed:

`<class id> <x-center> <y-center> <width> <height>` relative, i.e. [0, 1]

(`<class id>` is an enumeration of all classes. All values are separated by space. Objects are simply separated by a linebreak.)
Additional points of segmentation masks or a key-point structure are simply appended: `... <x1> <y1> <x2> <y2> ...` for segmentation masks or `... <x1> <y1> <v1> <x2> <y2> <v2> ...` for key-points (`<v?>` specifies if the point is visible).


You can also download the [MS COCO](https://cocodataset.org/) dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip).


Note that the folder contains a second typ of configuration files starting with **hyp.** This lists the hyperparameters to use when training the model. See below for details.
Check out the code in [train.py](./train.py) to get the ranges in which their values are evolved (look for a dictionary called `meta`, e.g. the pattern `meta = {`) or check out the *hyp.*-files to get a list of all hyperparameters.



## Miscellaneous
The folder [VisualizationTools](./VisualizationTools) inhabits two scripts that were used to generate a video of the augmented images. It is hardly relevant for any project unless you want to get a better gasp of your data.

When working with a small dataset, it is sometimes advisable to augment even the validation data. However, this should not happen at runtime in a dynamic way as it would obfuscate the metrics. So you may want to augment the validation data once in advance. Use the script [augment_validation_set.py](./augment_validation_set.py) for this.



## Acknowledgments

- Original authors: Chien-Yao Wang , Alexey Bochkovskiy, and Hong-Yuan Mark Liao
- Original code: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- Original paper: [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696), [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

## Notes
- currently the [albumentations](https://albumentations.ai/)-package is required (even if only [test.py](test.py) or [detect.py](detect.py) are used) as it is loaded with [utils.datasets.py](utils%2Fdatasets.py) anyway. Later to be separated.

## Author
 - max-scw

## Road map
- segmentation
- key point detection

## Status
active