import os
import re

from tqdm import tqdm
import numpy as np
import torch

LIMIT = 50000

class ExpandedDataset:

    def __init__(self, args, dataset, imagenet):
        self.dataset = dataset
        self.train_dataset = dataset.train_dataset
        self.imagenet = imagenet
        self.imagenet_train_dataset = imagenet.train_dataset
        self.shift = imagenet.num_class # should be num_class of the ImageNetSM
        self.img_out = []
        self.label_out = []
        self.expand()

    def expand(self):
        print("[Expanding] add ImageNet")
        for i in tqdm(np.arange(
            min(LIMIT, len(self.imagenet_train_dataset))
        )):
            images = self.imagenet_train_dataset[i]["images"]
            labels = self.imagenet_train_dataset[i]["labels"]
            self.img_out.append(
                images.tolist()
            )
            self.label_out.append(
                labels
            )
        print("[Expanding] add target_dataset")
        for j in tqdm(np.arange(
            min(LIMIT, len(self.train_dataset))
        )):
            images = self.train_dataset[j][0]
            labels = self.train_dataset[j][1] + self.shift
            self.img_out.append(
                images.tolist()
            )
            self.label_out.append(
                labels
            )
    def get(self):
        assert len(self.img_out) == len(self.label_out)
        random_idx = np.arange(len(self.img_out))
        np.random.shuffle(random_idx)
        images = torch.tensor(self.img_out)[random_idx]
        labels = torch.tensor(self.label_out)[random_idx]
        return images, labels






    