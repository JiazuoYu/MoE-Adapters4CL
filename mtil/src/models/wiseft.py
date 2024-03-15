import torch
import os

from .evaluation_fc import evaluate_fc
from .modeling import create_zeroshot_classifier_head


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


class WISE_FT:
    def __init__(self, args, zeroshot, finetuned) -> None:
        theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
        theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
        del zeroshot
        # print(theta_0.keys())
        # print(" ")
        # print(theta_1.keys())

        assert set(theta_0.keys()) == set(theta_1.keys())

        if args.fisher is None:
            fishers = None
        else:
            fisher_0_file, fisher_1_file = args.fisher
            fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
            fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
            fishers = fisher_0, fisher_1

        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.fishers = fishers
        self.fisher_floor = args.fisher_floor

    def __call__(self, alpha):
        theta = _merge(
            alpha, self.theta_0, self.theta_1, self.fishers, self.fisher_floor
        )
        return theta


def evaluate_wise_ft(args, zeroshot, finetuned, save=True):
    assert args.train_dataset is not None, "Please provide a train dataset to get embedding."

    if args.train_dataset:
        finetuned.classification_head = create_zeroshot_classifier_head(
            args, dataset=args.train_dataset
    )

    wise_ft = WISE_FT(args, zeroshot, finetuned)

    alphas = args.alpha
    for alpha in alphas:
        theta = wise_ft(alpha)
        finetuned.load_state_dict(theta)
        if save:
            finetuned.save(os.path.join(args.save, f"wise_ft_alpha={alpha:.3f}.pt"))
        evaluate_fc(finetuned, args)
