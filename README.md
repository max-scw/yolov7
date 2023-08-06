# Slim YOLOv7

Fork of the original [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) by [Kin-Yiu Wong](https://github.com/WongKinYiu). 

Supposed to present a concise version of the original code for greater usability. 
All credit must go to the original authors. This version was created out of the endeavor to understand the code in detail. It is due to my limited comprehension that part of the code was re-formatted, re-written, re-named or deleted to align with my understanding of the academic paper ([YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696), [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)), Python and Neural Networks.

Consider this project rather a "for dummies" fork.


## Comments and notes (on the original work)
### Idea (quick pass-through the paper [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696))
*Core idea*: "bag of freebies", i.e. invest at training time to reduce inference time => reduce about 40% parameters and 50% computation compared to state-of-the-art real-time object detector.

To explore the the "bag of freebies" (i.e. the hyper-parameters), an  evolution-inspired search is started (well, a grid search would indeed be exhaustive.)

### project structure
### Network configuration
**Network configuration** can be found in the [cfg](./cfg/) folder, e.g. [YOLOv7](./cfg/training/yolov7.yaml) or [tiny YOLOv7](./cfg/training/yolov7.yaml) which are the relevant structures. The YAML-written model structures are parsed when setting up the training (by the [yolo.py](./models/yolo.py) in the [models](./models/) folder.) This provides great flexibility when evaluating different model structures, e.g. to benchmark against baseline models (e.g. [YOLOv3](./cfg/baseline/yolov3.yaml), [YOLOv4](./cfg/baseline/yolov4-csp.yaml), [YOLOE](./cfg/baseline/yolor-e6.yaml) of the same authors, thus with the same backend "[Darknet](https://pjreddie.com/darknet/)") or evaluate scaled structures ([YOLOv7-W6](./cfg/training/yolov7-w6.yaml), [YOLOv7-E6](./cfg/training/yolov7-e6.yaml), [YOLOv7-D6](./cfg/training/yolov7-d6.yaml).)

### Data

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

(`<class id>` is an enumeration of all classes. All values are separated by space. Objects are simply separated by a linkebreak.)




You can also download the MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip).


Note that the folder contains a second typ of configuration files starting with **hyp.** This lists the hyperparamters to use when training the model. I recommend to use them right away. However, you can determine the perfect set of hyperparamters specific to your dataset by adding the `--evolve` when calling `python train.py`. This will trigger an evolutionary optimization of all hyperparamters, training the model multiple times (per default for 300 generations, adjustable by the `--evolve-generations` flag).

Check out the code in [train.py](./train.py) to get the ranges in which their values are evolved (look for a dictionary called `meta`, e.g. the pattern `meta = {`) or check out the *hyp.*-files to get a list of all hyperparameters.


### Deployment
The original work provides a nice integration for deploying the network primary to an [Nvidia Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) (an open-source project by Nvidia on [github](https://github.com/triton-inference-server/server) for a containerized server that distributes the workload of multiple models and requests on large Nvidia GPUs.) As CPUs perfectly suit my needs for inference, I barely had a look into this convenient deployment framework.  

There also exists the option to export the model as [Open Neural Network eXchange (ONNX)](https://onnx.ai/) format with [export.py](./export.py). I have experienced issued on the Non-Maximum Suppression algorithm, most likely due to my limited knowledge. However if there actually might be a patch needed, the original project is presumably the better address. Therefore, the deployment folder was not forked to this project.

### Miscellaneous
The folder [VisualizationTools](./VisualizationTools) inhabits two scripts that were used to generate a video of the augmented images. It is hardly relevant for any project unless you want to get a better gasp of your data.

When working with a small dataset, it is sometimes advisable to augment even the validations data. However, this should not happen at runtime in a dynamic way as it would obfuscate the metrics. So you may want to augment the validation data once in advance. Use the script [augment_validation_set.py](./augment_validation_set.py) for this.



## Installation
I recommend to use Python 3.9, install the required dependencies (to your virtual environment, I hope this goes without saying ;))
````shell
pip install -r requirements.txt
````
BTW, the model relies on [PyTorch](https://pytorch.org/). Check out how to export the trained model to ONNX format with [export.py](./export.py) to integrate it to other NN-frameworks (or without). 

## Usage
### Training

You can train a network with [train.py](./train.py) -- so far so obvious.
Example call:
````shell
python train.py --device 'cpu' --batch-size 16 --data data/myYOLO.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name my-yolov7 --hyp data/hyp.scratch.custom.yaml --epochs 300 --adam
````

This also provides already trained weights [`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) on the MS COCO dataset (**transfer learning**). Note that one should first train the head of the model if the head differs from the pre-trained dataset (e.g. different number of classes.) For this, freeze the body and train the network for some epochs `python train.py --freeze 60 ...` (check for how many layers the weights were transferred. This is printed on the console when setting up the network. For a *tiny YOLOv7* I have godf experiences with freezing the first XX layers.)
Afterwards, run the training a second time without freezing the layers and use the final weights of the previous run as input. This **fine-tuning** very much likely further improves accuracy. This two-stage training should require less than 1/10 of the time of training the model from only partly matching weights (doing transfer learning with a random head) and of course much much less than training everything from scratch.

Unfortunately, no weights were provided for the tiny version, which is the version I prefer.


All code can be accessed conveniently by command line commands and adjusted with extra arguments. Simply list all options with `python train.py --help`
```shell
>> python train.py --help
usage: train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--img-size IMG_SIZE [IMG_SIZE ...]] [--rect] [--resume [RESUME]] [--nosave] [--notest] [--noautoanchor] [--evolve] [--evolve-generations EVOLVE_GENERATIONS] [--bucket BUCKET] [--cache-images] [--image-weights] [--device DEVICE] [--multi-scale] [--single-cls] [--adam] [--sync-bn] [--local_rank LOCAL_RANK] [--workers WORKERS] [--project PROJECT] [--entity ENTITY] [--name NAME] [--exist-ok] [--quad] [--linear-lr] [--label-smoothing LABEL_SMOOTHING] [--save_period SAVE_PERIOD] [--save-best-n-checkpoints SAVE_BEST_N_CHECKPOINTS] [--freeze FREEZE [FREEZE ...]] [--no-augmentation] [--v5-metric] [--albumentations_probability ALBUMENTATIONS_PROBABILITY] [--export-training-images EXPORT_TRAINING_IMAGES] [--no-mosaic-augmentation]

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
  --bucket BUCKET       gsutil bucket
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
  --albumentations_probability ALBUMENTATIONS_PROBABILITY
                        Probability to apply data augmentation based on the albumentations package.
  --export-training-images EXPORT_TRAINING_IMAGES
                        Folder where to export the (augmented) training images to.
  --no-mosaic-augmentation
                        Do not apply mosaic augmentation.
```

There are a few minor extensions, e.g. for limiting the number of checkpoints to be stored (in order to avoid good old storage overflow). Most of them account for training much smaller datasets (than MS COCO), e.g. with only a few hundred examples. So the number of epochs is usually much larger and cluttering the hard drive with checkpoints actually a thread one needs to think about.
**The most interesting extension** though is a much more aggressive image augmentation triggered by the flag `--albumentations_probability`. Again, this is accounts for small dataset. One can set the probability for all of the augmentation methods (the higher the longer the training but the more resilent is the resulting model to changes in illumination.) The original work uses a probability of 5% (0.05), I have good experiences with >> 30%. (Note, when setting `--albumentations_probability=0.05` only the augmentation that was used in the original work is applied. My extension uses a few more methods of the [albumentations](https://albumentations.ai/) package.)

## Inference
Once you have a trained model, inference is done in [detect.py](./detect.py):

On video:
``` shell
python detect.py --weights best.pt --conf 0.25 --img-size 640 --source <path to video>
```

On image:
``` shell
python detect.py --weights best.pt --conf 0.25 --img-size 640 --source <path to image>
```

I've added a little watchdog system that runs the model every time a new image is placed in the folder that is monitored (e.g. if you have an ftp-server where new immages arrive). Check it out with
``` shell
python watchdog_service.py --weights best.pt --conf-thres 0.25 --img-size 640 --folder <path to folder>
```

## Acknowledgments

- Original authors: Chien-Yao Wang , Alexey Bochkovskiy, and Hong-Yuan Mark Liao
- Original code: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- Original paper: [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696), [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)



## Author
 - max-scw

## Road map
- add key point detection

## Status
active