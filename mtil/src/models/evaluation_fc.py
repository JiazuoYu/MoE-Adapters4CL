import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .. import datasets, templates, utils
from ..datasets.common import get_dataloader, maybe_dictionarize
from .modeling import create_zeroshot_classifier_head


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

    metrics = {}
    metrics["top1"] = top1

    return metrics


def evaluate_fc(image_classifier, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    old_head = image_classifier.classification_head

    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        if args.dataset_shift:
            image_classifier.classification_head = create_zeroshot_classifier_head(
                args, dataset=dataset_name
            )

        results = eval_single_dataset(image_classifier, dataset, args)

        if "top1" in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if "worst" in key or "f1" in key.lower() or "pm0" in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ":" + key] = val

    image_classifier.classification_head = old_head

    # Save results
    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, "a+") as f:
            f.write(json.dumps(info) + "\n")
        print(f"Results saved to {args.results_db}.")
    else:
        print("Results not saved (to do so, use --results_db to specify a path).")

    return info
