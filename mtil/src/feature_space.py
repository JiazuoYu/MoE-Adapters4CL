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
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def visualize(models):
    pass
    

@torch.no_grad()
def make_dataframe(models, args):
    storage = None
    mmy = 0
    class_col = []
    # add high-dim features
    for model_ckpt in models:
        model, _, val_preprocess = clip.load(args.model, jit=False)
        if model_ckpt == "CLIP":
            print("[Evaluation] CLIP")
            model = model
        elif model_ckpt == "exp_aircraft_ft":
            print("[Evaluation] Finetuned")
            utils.torch_load(model, f"ckpt/{model_ckpt}/Aircraft.pth")
        else:
            print("[Evaluation] Method")
            utils.torch_load(model, f"ckpt/{model_ckpt}/SUN397.pth")

        dataset_class = getattr(datasets, "Aircraft")
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        model.eval()
        image_enc = None
        dataloader = get_dataloader(
            dataset, is_train=False, args=args, image_encoder=image_enc
        )
        for i, data in enumerate(tqdm(dataloader)):
            data = maybe_dictionarize(data)
            images = data["images"].cuda()
            target = data["labels"].cuda()

            image_features = model.encode_image(images)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            class_col += list(target.cpu().detach().numpy())
            if mmy == 0:
                storage = image_features.cpu().detach().numpy()
                mmy += 1
            else:
                storage_cur = image_features.cpu().detach().numpy()
                storage = np.concatenate((storage, storage_cur), axis=0)
    
    # change to low-dim
    print("[t-SNE] dimensional reduction")
    storage = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=3).fit_transform(storage)    
    
    # add label
    storage = pd.DataFrame(storage)
    method_col = []
    for method in ["CLIP", "Finetune", "WC", "LwF", "ZSCL"]:
        method_col += [f"{method}" for _ in np.arange(len(storage)/5)]
    storage["method"] = method_col
    storage["class"] = class_col
    return storage




if __name__ == "__main__":
    args = parse_arguments()
    models = ["CLIP", "exp_aircraft_ft", "exp_wc", "exp_lwf", "exp_zscl"]

    df = pd.DataFrame()
    storage = make_dataframe(models, args)

    print("[Plotting] feature_space")
    g = sns.FacetGrid(storage, col="method", hue="class")
    g.map(sns.scatterplot, 0, 1, alpha=0.5)
    plt.savefig("feature_space.png")

