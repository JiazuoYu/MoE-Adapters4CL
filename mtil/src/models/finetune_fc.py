import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import datasets, utils
from ..datasets.common import get_dataloader, maybe_dictionarize
from .evaluation_fc import evaluate_fc
from .modeling import create_image_classifier, create_zeroshot_classifier_head
from .helpers import paired_loss_new

def finetune_fc(args):
    if args.load is not None:
        print(f"[Start Continuous Learning] on {args.train_dataset}")
        model = create_image_classifier(args, setnone=True)
        utils.torch_load(model, args.load)
        model.classification_head = create_zeroshot_classifier_head(args, dataset=None)
    else:
        model = create_image_classifier(
            args, initialize=args.fc_init, setnone=args.fc_setnone
        )

    train_preprocess = model.train_preprocess
    model.process_images = True

    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )
    input_key = "images"
    image_enc = None

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval

    # get params
    if args.train_mode == "image-fc-fixed":
        for p in model.classification_head.parameters():
            p.requires_grad = False
        print("[Training mode] Image Encoder + FC (fixed)")
    else:
        print("[Training mode] Image Encoder + FC")
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # Ours
    if args.method == "ours":
        # (Anchor Model) get zeroshot model
        print("Start Continuous Learning with Trio")
        zeroshot = create_image_classifier(args)
        zeroshot = zeroshot.cuda()
        zeroshot = torch.nn.DataParallel(zeroshot, device_ids=devices)
        zeroshot.eval()

        # (Anchor Dataset) get anchor dataset
        anchor_class = getattr(datasets, args.anchor_dataset)
        print(f"[Anchor Dataset] {args.anchor_dataset}")
        anchor = anchor_class(
            train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )

    for iteration in tqdm(range(total_iterations + 1)):
        # evaluation
        if iteration % eval_iterations == 0:
            eval_results = evaluate_fc(model.module, args)

        # training (dataset)
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)
            if args.method == "ours":
                anchor_iter = iter(anchor.train_loader)

        # training (backpropagation)
        model.train()
        scheduler(iteration)

        # prepare data
        batch = next(data_iter)
        batch = maybe_dictionarize(batch)
        inputs = batch[input_key].cuda()
        labels = batch["labels"].cuda()

        if args.method == "ours":
            try:
                anchor_batch = next(anchor_iter)
            except:
                anchor = anchor_class(
                    train_preprocess,
                    location=args.data_location,
                    batch_size=args.batch_size,
                )
                anchor_iter = iter(anchor.train_loader)
                anchor_batch = next(anchor_iter)
            anchor_inputs = anchor_batch[input_key].cuda()

            ### compute anchor loss
            out_new, features_new = model(anchor_inputs)
            with torch.no_grad():
                out_old, features_old = zeroshot(anchor_inputs)
            loss_anchor = paired_loss_new(out_new, out_old)

            ### compute classification loss
            logits, _ = model(inputs)
            loss_classification = F.cross_entropy(logits, labels)
            ### compute final loss
            loss = torch.add(loss_classification, loss_anchor, alpha=1.5)
        else:
            # get new loss
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits, labels, label_smoothing=args.ls)

        # update
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if iteration % eval_iterations == 0:
            print("Loss:", loss.item())

    # Saving model
    if args.save is not None:
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(model.module, path)
