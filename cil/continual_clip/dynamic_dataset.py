import os
import re

import torch
from tqdm import tqdm
import numpy as np
# from torchvision import datasets
from . import datasets, utils
# import clip.clip as clip
import clip
import numpy as np

class DynamicDataset():

    def __init__(self, cfg):
        self.ref_database = {}  # all data key = dataset_name; value is 4D tensor (num_image, 3, 224, 224)
        self.ref_names = [] # collect the name of the dataset 
        self.ref_model, _, self.test_preprocess = clip.load(cfg.model_name, jit=False)
        self.cur_dataset = None
        self.memory_size = 5000
        self.batch_id = 0

    def update(self, dataset, load):
        # load is a model directly
        self.cur_dataset = dataset
        if not load: # first round USELESS FOR CICL
            new_dataset = self.getNewDataset()
            self.ref_database[self.cur_dataset] = new_dataset[:self.memory_size]
            self.ref_names.append(self.cur_dataset)
        else: # other rounds
            self.ref_model = load
            self.reduceExampleSet()
            self.constructExampleSet()

    def reduceExampleSet(self):
        print("Reducing Example Set")
        K, t = self.memory_size, len(self.ref_names)+1
        m = K // t
        for dataset in self.ref_names:
            self.ref_database[dataset] = self.ref_database[dataset][:m]

    def constructExampleSet(self):
        # breakpoint()
        print("Constructing Example Set")
        self.ref_names.append(self.batch_id)
        new_dataset = torch.tensor(self.getNewDataset())
        image_feature = []
        num = new_dataset.shape[0]

        print("[Constructing] Calculating Distance")
        for ndx in tqdm(np.arange(num)):
            img = torch.unsqueeze(new_dataset[ndx], dim=0)
            img = img.cuda()
            img_feature = self.ref_model(img, None)
            image_feature.append(img_feature.cpu().detach().tolist())
        image_feature = torch.tensor(image_feature)
        image_feature = torch.squeeze(image_feature, dim=1)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        image_feature = np.array(image_feature.cpu().detach())
        image_feature_average = image_feature.mean(axis=0)

        K, t = self.memory_size, len(self.ref_names)
        m = K - K // t
        update_dataset = []
        if not m:
            m = self.memory_size
        cur_embedding_sum = None
        print("[Constructing] Collecting Examples")
        for k in tqdm(np.arange(min(m, len(image_feature)))):
            if not k:
                index = np.argmin(
                    np.sum((image_feature_average - image_feature)**2, axis=1)
                )
                cur_embedding_sum = image_feature[index]
                update_dataset.append((new_dataset.cpu())[index].tolist())
                image_feature = np.delete(image_feature, index, axis=0)
            else:
                index = np.argmin(
                    np.sum((
                        image_feature_average - (1/(k+1))*(image_feature + cur_embedding_sum)
                    )**2, axis=1)
                )
                cur_embedding_sum += image_feature[index]
                update_dataset.append((new_dataset.cpu())[index].tolist())
                image_feature = np.delete(image_feature, index, axis=0)
        
        self.ref_database[self.batch_id] = update_dataset
        print("finishing current task", self.batch_id)
        self.batch_id = self.batch_id + 1

    def getNewDataset(self):
        samples = [] 
        count = 0
        for sample in tqdm(self.cur_dataset):
            if count == 10000:
                return samples
            count += 1
            samples.append(sample[0].tolist())
        return samples   

    def get(self):
        print("Getting Reference Images")
        value = list(self.ref_database.values())
        out = []
        for i in tqdm(value):
            out += i       
        return torch.tensor(out)
    