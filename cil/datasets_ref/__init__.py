# WILDS
# CIFAR
from .cifar10 import CIFAR101, CIFAR102

# Small
from .collections import (
    CIFAR10,
    CIFAR100,
    DTD,
    MNIST,
    SUN397,
    Aircraft,
    Caltech101,
    EuroSAT,
    Flowers,
    Food,
    OxfordPet,
    StanfordCars,
)
from .fmow import FMOW, FMOWID, FMOWOOD

# ImageNet
from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenet_small import ImageNetSM
from .imagenet_sub import ImageNetSUB
from .imagenet_subclass import ImageNetSC
from .imagenet_vid_robust import ImageNetVidRobust
from .imagenetv2 import ImageNetV2
from .iwildcam import (
    IWildCam,
    IWildCamID,
    IWildCamIDNonEmpty,
    IWildCamOOD,
    IWildCamOODNonEmpty,
)
from .joint import Joint

# Random Noise
from .noise import Noise
from .objectnet import ObjectNet
from .ytbb_robust import YTBBRobust

# Experimental datasets
dataset_list = [
    Aircraft,
    Caltech101,
    CIFAR10,
    CIFAR100,
    DTD,
    EuroSAT,
    Flowers,
    Food,
    MNIST,
    OxfordPet,
    StanfordCars,
    SUN397,
]

def show_datasets():
    print("Total: ", len(dataset_list))
    print("Dataset: (train_len, test_len, num_classes)")
    for dataset in dataset_list:
        d = dataset(None)
        print(f"{d.name}: ", d.stats())
        for i in range(3):
            print(f"T[{i}]: ", d.template(d.classnames[i]))
        

from .cc import conceptual_captions
