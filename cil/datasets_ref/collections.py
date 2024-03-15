import os
import re

import torch
from torchvision import datasets

from .cifar10 import CIFAR10 as cifar10, CIFAR100 as cifar100


def underline_to_space(s):
    return s.replace("_", " ")


class ClassificationDataset:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        self.name = "classification_dataset"
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        if batch_size_eval is None:
            self.batch_size_eval = batch_size
        else:
            self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.append_dataset_name_to_template = append_dataset_name_to_template

        self.train_dataset = self.test_dataset = None
        self.train_loader = self.test_loader = None
        self.classnames = None
        self.templates = None

    def build_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def stats(self):
        L_train = len(self.train_dataset)
        L_test = len(self.test_dataset)
        N_class = len(self.classnames)
        return L_train, L_test, N_class

    @property
    def template(self):
        if self.append_dataset_name_to_template:
            return lambda x: self.templates[0](x)[:-1] + f", from dataset {self.name}]."
        return self.templates[0]

    def process_labels(self):
        self.classnames = [underline_to_space(x) for x in self.classnames]

    def split_dataset(self, dataset, ratio=0.8):
        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, test_dataset
    
    @property
    def class_to_idx(self):
        return {v: k for k, v in enumerate(self.classnames)}


class Aircraft(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "aircraft"
        self.train_dataset = datasets.FGVCAircraft(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.FGVCAircraft(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of aircraft.",
            lambda c: f"a photo of the {c}, a type of aircraft.",
        ]

    # def process_labels(self):
    #     label = self.classnames
    #     for i in range(len(label)):
    #         if label[i].startswith("7"):
    #             label[i] = "Boeing " + label[i]
    #         elif label[i].startswith("An") or label[i].startswith("ATR"):
    #             pass
    #         elif label[i].startswith("A"):
    #             label[i] = "Airbus " + label[i]


class Caltech101(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "caltech101"
        dataset = datasets.Caltech101(
            self.location, download=True, transform=self.preprocess
        )
        self.classnames = dataset.categories

        train_dataset, test_dataset = self.split_dataset(dataset)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()

        self.classnames = [
            "off-center face",
            "centered face",
            "leopard",
            "motorbike",
            "accordion",
            "airplane",
            "anchor",
            "ant",
            "barrel",
            "bass",
            "beaver",
            "binocular",
            "bonsai",
            "brain",
            "brontosaurus",
            "buddha",
            "butterfly",
            "camera",
            "cannon",
            "side of a car",
            "ceiling fan",
            "cellphone",
            "chair",
            "chandelier",
            "body of a cougar cat",
            "face of a cougar cat",
            "crab",
            "crayfish",
            "crocodile",
            "head of a  crocodile",
            "cup",
            "dalmatian",
            "dollar bill",
            "dolphin",
            "dragonfly",
            "electric guitar",
            "elephant",
            "emu",
            "euphonium",
            "ewer",
            "ferry",
            "flamingo",
            "head of a flamingo",
            "garfield",
            "gerenuk",
            "gramophone",
            "grand piano",
            "hawksbill",
            "headphone",
            "hedgehog",
            "helicopter",
            "ibis",
            "inline skate",
            "joshua tree",
            "kangaroo",
            "ketch",
            "lamp",
            "laptop",
            "llama",
            "lobster",
            "lotus",
            "mandolin",
            "mayfly",
            "menorah",
            "metronome",
            "minaret",
            "nautilus",
            "octopus",
            "okapi",
            "pagoda",
            "panda",
            "pigeon",
            "pizza",
            "platypus",
            "pyramid",
            "revolver",
            "rhino",
            "rooster",
            "saxophone",
            "schooner",
            "scissors",
            "scorpion",
            "sea horse",
            "snoopy (cartoon beagle)",
            "soccer ball",
            "stapler",
            "starfish",
            "stegosaurus",
            "stop sign",
            "strawberry",
            "sunflower",
            "tick",
            "trilobite",
            "umbrella",
            "watch",
            "water lilly",
            "wheelchair",
            "wild cat",
            "windsor chair",
            "wrench",
            "yin and yang symbol",
        ]

        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a painting of a {c}.",
            lambda c: f"a plastic {c}.",
            lambda c: f"a sculpture of a {c}.",
            lambda c: f"a sketch of a {c}.",
            lambda c: f"a tattoo of a {c}.",
            lambda c: f"a toy {c}.",
            lambda c: f"a rendition of a {c}.",
            lambda c: f"a embroidered {c}.",
            lambda c: f"a cartoon {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"a plushie {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"art of a {c}.",
            lambda c: f"graffiti of a {c}.",
            lambda c: f"a drawing of a {c}.",
            lambda c: f"a doodle of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a painting of the {c}.",
            lambda c: f"the plastic {c}.",
            lambda c: f"a sculpture of the {c}.",
            lambda c: f"a sketch of the {c}.",
            lambda c: f"a tattoo of the {c}.",
            lambda c: f"the toy {c}.",
            lambda c: f"a rendition of the {c}.",
            lambda c: f"the embroidered {c}.",
            lambda c: f"the cartoon {c}.",
            lambda c: f"the {c} in a video game.",
            lambda c: f"the plushie {c}.",
            lambda c: f"the origami {c}.",
            lambda c: f"art of the {c}.",
            lambda c: f"graffiti of the {c}.",
            lambda c: f"a drawing of the {c}.",
            lambda c: f"a doodle of the {c}.",
        ]


class MNIST(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "mnist"
        self.train_dataset = datasets.MNIST(
            self.location, train=True, download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.MNIST(
            self.location, train=False, download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of the number: "{c}".',
        ]


class CIFAR10(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar10"
        dataset = cifar10(preprocess=self.preprocess, location=self.location)

        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset
        self.build_dataloader()
        self.classnames = dataset.classnames
        self.process_labels()
        self.templates = dataset.template


class CIFAR100(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar100"
        dataset = cifar100(preprocess=self.preprocess, location=self.location)

        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset
        self.build_dataloader()
        self.classnames = dataset.classnames
        self.process_labels()
        self.templates = dataset.template


class DTD(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "dtd"
        self.train_dataset = datasets.DTD(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.DTD(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of a {c} texture.',
            lambda c: f'a photo of a {c} pattern.',
            lambda c: f'a photo of a {c} thing.',
            lambda c: f'a photo of a {c} object.',
            lambda c: f'a photo of the {c} texture.',
            lambda c: f'a photo of the {c} pattern.',
            lambda c: f'a photo of the {c} thing.',
            lambda c: f'a photo of the {c} object.',
        ]


class EuroSAT(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "eurosat"
        dataset = datasets.EuroSAT(
            self.location, download=True, transform=self.preprocess
        )
        train_dataset, test_dataset = self.split_dataset(dataset)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()

        self.classnames = [
            "annual crop land",
            "forest",
            "brushland or shrubland",
            "highway or road",
            "industrial buildings or commercial buildings",
            "pasture land",
            "permanent crop land",
            "residential buildings or homes or apartments",
            "river",
            "lake or sea",
        ]

        self.templates = [
            lambda c: f"a centered satellite photo of {c}.",
            lambda c: f"a centered satellite photo of a {c}.",
            lambda c: f"a centered satellite photo of the {c}.",
        ]

    def process_labels(self):
        super().process_labels()
        self.classnames = [re.sub(r"(\w)([A-Z])", r"\1 \2", x) for x in self.classnames]


class Flowers(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "flowers"
        self.train_dataset = datasets.Flowers102(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.Flowers102(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink-yellow dahlia",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of flower.",
        ]


class Food(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "food"
        self.train_dataset = datasets.Food101(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.Food101(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of food.",
        ]


class OxfordPet(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "oxford pet"
        self.train_dataset = datasets.OxfordIIITPet(
            self.location, split="trainval", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.OxfordIIITPet(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of pet.",
        ]


class StanfordCars(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "stanford cars"
        self.train_dataset = datasets.StanfordCars(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.StanfordCars(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of car.",
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"i love my {c}!",
            lambda c: f"a photo of my dirty {c}.",
            lambda c: f"a photo of my clean {c}.",
            lambda c: f"a photo of my new {c}.",
            lambda c: f"a photo of my old {c}.",
        ]


class SUN397(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "sun397"
        dataset = datasets.SUN397(
            self.location, download=True, transform=self.preprocess
        )
        train_dataset, test_dataset = self.split_dataset(dataset)
        self.classnames = dataset.classes

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
