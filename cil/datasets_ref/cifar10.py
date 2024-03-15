import os

import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
from torchvision.datasets import VisionDataset

cifar10_template = [
    lambda c : f'a photo of a {c}.',
    lambda c : f'a blurry photo of a {c}.',
    lambda c : f'a black and white photo of a {c}.',
    lambda c : f'a low contrast photo of a {c}.',
    lambda c : f'a high contrast photo of a {c}.',
    lambda c : f'a bad photo of a {c}.',
    lambda c : f'a good photo of a {c}.',
    lambda c : f'a photo of a small {c}.',
    lambda c : f'a photo of a big {c}.',
    lambda c : f'a photo of the {c}.',
    lambda c : f'a blurry photo of the {c}.',
    lambda c : f'a black and white photo of the {c}.',
    lambda c : f'a low contrast photo of the {c}.',
    lambda c : f'a high contrast photo of the {c}.',
    lambda c : f'a bad photo of the {c}.',
    lambda c : f'a good photo of the {c}.',
    lambda c : f'a photo of the small {c}.',
    lambda c : f'a photo of the big {c}.',
]


DOWNLOAD = True
cifar_classnames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class CIFAR10:
    def __init__(
        self,
        preprocess,
        location="./data",
        batch_size=128,
        num_workers=16,
        classnames=None,
    ):

        self.train_dataset = PyTorchCIFAR10(
            root=location, download=DOWNLOAD, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = PyTorchCIFAR10(
            root=location, download=DOWNLOAD, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.test_dataset.classes
        self.template = cifar10_template


def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x


class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(
            root=None, transform=transform, target_transform=target_transform
        )
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)


class CIFAR101:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
        classnames=None,
    ):

        data_root = os.path.join(location, "CIFAR-10.1")
        data = np.load(
            os.path.join(data_root, "cifar10.1_v6_data.npy"), allow_pickle=True
        )
        labels = np.load(
            os.path.join(data_root, "cifar10.1_v6_labels.npy"), allow_pickle=True
        )

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.train_loader = None
        self.location = location

        self.test_dataset = BasicVisionDataset(
            images=data,
            targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.classnames = cifar_classnames
        self.template = cifar10_template


class CIFAR102:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=16,
        classnames=None,
    ):

        train_data = np.load(
            os.path.join(location, "CIFAR-10.2", "cifar102_train.npy"),
            allow_pickle=True,
        ).item()
        test_data = np.load(
            os.path.join(location, "CIFAR-10.2", "cifar102_test.npy"), allow_pickle=True
        ).item()

        train_data_images = train_data["images"]
        train_data_labels = train_data["labels"]

        test_data_images = test_data["images"]
        test_data_labels = test_data["labels"]

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.test_dataset = BasicVisionDataset(
            images=test_data_images,
            targets=torch.Tensor(test_data_labels).long(),
            transform=preprocess,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.train_dataset = BasicVisionDataset(
            images=train_data_images,
            targets=torch.Tensor(train_data_labels).long(),
            transform=preprocess,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.classnames = cifar_classnames
        self.template = cifar10_template


class CIFAR100:
    def __init__(
        self,
        preprocess,
        location="./data",
        batch_size=128,
        num_workers=16,
        classnames=None,
    ):

        self.train_dataset = PyTorchCIFAR100(
            root=location, download=DOWNLOAD, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=DOWNLOAD, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.test_dataset.classes
        self.template = cifar10_template
