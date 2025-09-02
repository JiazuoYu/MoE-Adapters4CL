from .args import parse_arguments
from .models.modeling import create_image_classifier
from . import utils
import clip
import os
from . import datasets
from .models.modeling import create_zeroshot_classifier_head
from .datasets.common import get_dataloader, maybe_dictionarize
import torch
from tqdm import tqdm
import numpy as np

def eval_single_dataset(image_classifier, dataset, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = "features"
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = "images"
        image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )
    batched_data = enumerate(dataloader)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0

        for i, data in tqdm(batched_data):
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data["labels"].to(device)
            logits, feature = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n
    print(f"[accuracy] {top1:4f}")
    print(" ")

    metrics = {}
    metrics["top1"] = top1

    return metrics


def evaluate_fc(image_classifier, dataset):

    info = vars(args)
    old_head = image_classifier.classification_head

    for i, dataset_name in enumerate(dataset):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        image_classifier.classification_head = create_zeroshot_classifier_head(
            args, dataset=dataset_name
        )

        results = eval_single_dataset(image_classifier, dataset, args)

    image_classifier.classification_head = old_head

    return info

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model):
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    for classname in classnames:
        texts = [template(classname) for template in templates]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset_bl(image_classifier, dataset, args, RECORD):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights)

    print(f"Top-1 accuracy: {top1:.2f}")
    RECORD.append(top1)

def evaluate_bl(image_classifier, datasets1, val_preprocess, RECORD):

    for i, dataset_name in enumerate(datasets1):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        eval_single_dataset_bl(image_classifier, dataset, args, RECORD)

def evaluate(dictionary, agrs, fc, RECORD):
    for i in dictionary:
        model_ckpt = i[0]
        datasets = i[1]
        if fc:
            print(f"using checkpoint {model_ckpt}")
            model = create_image_classifier(args, setnone=True)
            utils.torch_load(model, model_ckpt)
            evaluate_fc(model, datasets)
        else:
            model, _, val_preprocess = clip.load(args.model, jit=False)
            utils.torch_load(model, model_ckpt)
            evaluate_bl(model, datasets, val_preprocess, RECORD)
    return 
        
if __name__ == "__main__":
    args = parse_arguments()
    NAMES_ARRAY =  ["exp_zscl"]

    for NAMES in NAMES_ARRAY:
        # NAMES = "exp_05"
        RECORD = []
        fc = False
        # dictionary = [
        #     [f"ckpt/{NAMES}/Aircraft.pth", ["Aircraft", "Caltech101"]],
        #     [f"ckpt/{NAMES}/Caltech101.pth", ["Caltech101", "CIFAR100"]],
        #     # [f"ckpt/{NAMES}/CIFAR10.pth", ["CIFAR10", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/CIFAR100.pth", ["CIFAR100", "DTD"]],
        #     [f"ckpt/{NAMES}/DTD.pth", ["DTD", "EuroSAT"]],
        #     [f"ckpt/{NAMES}/EuroSAT.pth", ["EuroSAT", "Flowers"]],
        #     [f"ckpt/{NAMES}/Flowers.pth", ["Flowers", "Food"]],
        #     [f"ckpt/{NAMES}/Food.pth", ["Food", "MNIST"]],
        #     [f"ckpt/{NAMES}/MNIST.pth", ["MNIST", "OxfordPet"]],
        #     [f"ckpt/{NAMES}/OxfordPet.pth", ["OxfordPet", "StanfordCars"]],
        #     [f"ckpt/{NAMES}/StanfordCars.pth", ["StanfordCars", "SUN397"]],
        #     [f"ckpt/{NAMES}/SUN397.pth", ["Aircraft", "Caltech101", "CIFAR10", "CIFAR100", "DTD", "EuroSAT", "Flowers",
        #                                         "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
        # ]
        # dictionary = [
        #     [f"ckpt/{NAMES}/CIFAR10.pth", ["CIFAR10", "StanfordCars"]],
        #     [f"ckpt/{NAMES}/StanfordCars.pth", ["StanfordCars", "Food"]],
        #     [f"ckpt/{NAMES}/Food.pth", ["Food", "MNIST"]],
        #     [f"ckpt/{NAMES}/MNIST.pth", ["MNIST", "OxfordPet"]],
        #     [f"ckpt/{NAMES}/OxfordPet.pth", ["OxfordPet", "Flowers"]],
        #     [f"ckpt/{NAMES}/Flowers.pth", ["Flowers", "SUN397"]],
        #     [f"ckpt/{NAMES}/SUN397.pth", ["SUN397", "Aircraft"]],
        #     [f"ckpt/{NAMES}/Aircraft.pth", ["Aircraft", "Caltech101"]],
        #     [f"ckpt/{NAMES}/Caltech101.pth", ["Caltech101", "DTD"]],
        #     [f"ckpt/{NAMES}/DTD.pth", ["DTD", "EuroSAT"]],
        #     [f"ckpt/{NAMES}/EuroSAT.pth", ["EuroSAT", "CIFAR100"]],
        #     [f"ckpt/{NAMES}/CIFAR100.pth", ["CIFAR10", "StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
        #                                         "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]],
        # ]
        dictionary = [
            [f"ckpt/{NAMES}/SUN397.pth", ["Aircraft", "Caltech101", "CIFAR10", "CIFAR100", "DTD", "EuroSAT", "Flowers",
                                                "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]],
        ]
        evaluate(dictionary, args, fc, RECORD)
        RECORD = np.array(RECORD)
        outfile = os.path.join("./results", f"{NAMES}.npy")
        np.save(outfile, RECORD)
        print(" ")
        print(" ")
        print(" ")

