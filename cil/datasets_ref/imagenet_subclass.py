import os

import numpy as np
import torch

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames
from ..templates.openai_imagenet_template import openai_imagenet_template

SUBCLASS = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107,
    108, 110,
    113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
    308, 309,
    310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397,
    400, 401,
    402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488,
    492, 496,
    514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614,
    626, 627,
    640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773,
    774, 776,
    779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870,
    879, 8801,
    888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980,
    981, 984,
    986, 987, 988]

SUBCLASS = np.arange(1000)[:100].tolist()


class ImageNetSC:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.template = openai_imagenet_template

        self.populate_train()
        self.populate_test()
    
    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess)

        
        samples  = []
        classes = []
        targets = []
        dic = self.train_dataset.class_to_idx
        for cla in self.train_dataset.classes:
            if dic[cla] in SUBCLASS:
                classes.append(cla)

        for i in np.arange(len(self.train_dataset.samples)):
            sample = self.train_dataset.samples[i]
            target = self.train_dataset.targets[i]
            if sample[1] in SUBCLASS: ##sample[1] == target
                samples.append(sample)
                targets.append(target)
        
        self.train_dataset.classes = classes
        self.train_dataset.samples = samples
        self.train_dataset.targets = targets

        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'