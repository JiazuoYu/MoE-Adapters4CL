import os

import numpy as np
import torch
import torchvision.transforms as T

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames
from ..templates.openai_imagenet_template import openai_imagenet_template

# num_class = 100
MEAN = (0.48145446, 0.4578275, 0.40821073)
VAR = (0.26862954, 0.26130258, 0.27577711)

class Joint:
# class Noise: ### noise_data_1005
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess_noise = T.Compose([T.ToTensor(), T.Normalize(MEAN, VAR)])
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_class = 10 #############################################
        self.num_pic = 100 ############################################# twice!!!
        assert self.num_pic <= 150, "please provide less than 150 images per class"

        # self.classnames = get_classnames(classnames)[:num_class]
        self.classnames = get_classnames(classnames)[:self.num_class]
        self.template = openai_imagenet_template

        self.populate_train()
        self.populate_test()
    
    def populate_train(self):
        SUBCLASS = np.arange(1000)[:self.num_class].tolist()
        ImageNetDir = os.path.join(self.location, self.name_ImageNet())
        NoiseDir = os.path.join(self.location, self.name_Noise())

        self.train_dataset = ImageFolderWithPaths(
            ImageNetDir,
            transform=self.preprocess)

        self.noise_dataset = ImageFolderWithPaths(
            NoiseDir,
            transform=self.preprocess_noise)  

        samples  = []
        targets = []
        classes = []
        storage = {}
        dic = self.train_dataset.class_to_idx
        noise_dic = self.noise_dataset.class_to_idx
        
        ### ImageNet
        for cla in self.train_dataset.classes:
            if dic[cla] in SUBCLASS:
                classes.append(cla)

        for i in np.arange(len(self.train_dataset.samples)):
            sample = self.train_dataset.samples[i]
            target = self.train_dataset.targets[i]
            if sample[1] in SUBCLASS:
                if sample[1] not in storage.keys():
                    storage[sample[1]] = 1
                    samples.append(sample)
                    targets.append(target)
                else: 
                    if storage[sample[1]] < self.num_pic/2:
                        samples.append(sample)
                        targets.append(target)
                        storage[sample[1]] += 1

        ### Noise
        for cla in self.noise_dataset.classes:
            if noise_dic[cla] in SUBCLASS:
                classes.append(cla)

        for i in np.arange(len(self.noise_dataset.samples)):
            sample = self.noise_dataset.samples[i]
            target = self.noise_dataset.targets[i]
            if sample[1] in SUBCLASS:
                if sample[1] not in storage.keys():
                    storage[sample[1]] = 1
                    samples.append(sample)
                    targets.append(target)
                    print("error")
                    breakpoint()
                else: 
                    if storage[sample[1]] < self.num_pic:
                        samples.append(sample)
                        targets.append(target)
                        storage[sample[1]] += 1        



        self.train_dataset.classes = classes
        self.train_dataset.samples = samples
        self.train_dataset.targets = targets  

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def populate_test(self):
        # self.test_dataset = self.get_test_datasset()
        self.test_loader = torch.utils.data.DataLoader(
            self.train_dataset,
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

    # def name(self):
    #     return "joint/train"
    def name_ImageNet(self):
        return "imagenet/train"

    def name_Noise(self):
        return 'noise_data_1000100'
            

