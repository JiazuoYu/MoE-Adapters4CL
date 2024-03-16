import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    # hyper parameters
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-size-eval", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--warmup_length", type=int, default=100)
    parser.add_argument("--beta2", type=float, default=0.999)

    # logging setting
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--loss-interval", type=int, default=1000)
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--eval-only", action="store_true")

    # exp setting
    parser.add_argument(
        "--method",
        type=str,
        default="finetune",
        choices=["finetune", "lwf", "ZSCL", "icarl"],
        help="Method to use.",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="whole",
        choices=["whole", "text", "image", "image-fc", "image-fc-fixed", "fc", "adapter"],
        help="Train mode to use.",
    )
    parser.add_argument("--data-location", type=str, default="./data")
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--text-datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--template", type=str, default=None)

    # save & load
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--load_federate", default=None, type=lambda x: x.split(","))
    parser.add_argument("--load_autochooser", type=str, default=None)  # the checkpoint of autochooser

    # model control for image-fc branch
    parser.add_argument("--fair", action="store_true")
    parser.add_argument("--we", action="store_true")
    parser.add_argument("--we_wise", action="store_true")
    parser.add_argument("--we_wise_alpha", type=float, default=0.98, help="wise_ft_alpha")
    parser.add_argument("--moving_avg", action="store_true")
    parser.add_argument("--avg_freq", type=int, default=100)
    parser.add_argument("--mv_avg_decay", type=float, default=0.999)
    parser.add_argument(
        "--mv_avg_model",
        type=str,
        default="n",
        choices=["n", "t", "zeroshot"],
        help="moving_avg_model to use.",
    )
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument(
        "--fc-init", action="store_true", help="Whether to reinitialize the model."
    )
    parser.add_argument(
        "--fc-setnone", action="store_true", help="Whether to shift the dataset."
    )
    parser.add_argument(
        "--dataset-shift", action="store_true", help="Whether to shift the dataset."
    )
    parser.add_argument("--n_class", type=int, default=10, help="Number of classes.")

    # ZSCL
    parser.add_argument(
        "--ref_wise_alpha", type=float, default=0.8, help="WiSE zeroshot reference"
    )
    parser.add_argument(
        "--ref-wise",
        default=False,
        action="store_true",
        help="WiSE zeroshot reference",
    )
    parser.add_argument(
        "--ref-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument("--ref-model", type=str, default=None)
    parser.add_argument(
        "--ref-sentences",
        default=None,
        help="For fine tuning or linear probe, which dataset's template and classname to train on",
    )
    parser.add_argument(
        "--T", type=float, default=2.0, help="Temperature for distillation loss"
    )
    parser.add_argument("--num", type=float, default=64)

    # --------- #
    # iCaRL
    parser.add_argument("--dataset_order", default=None, type=lambda x: x.split(","))
    parser.add_argument("--memory_size", type=int, default=10000)

    # --------- #
    # others
    parser.add_argument(
        "--weight_adjust",
        default=False,
        action="store_true",
        help="adjust",
    )
    parser.add_argument(
        "--feature_mse",
        default=False,
        action="store_true",
        help="feature_mse",
    )
    parser.add_argument(
        "--image_loss",
        default=False,
        action="store_true",
        help="image_loss",
    )
    parser.add_argument(
        "--text_loss",
        default=False,
        action="store_true",
        help="text_loss",
    )
    parser.add_argument(
        "--ablation_loss_2",
        default=False,
        action="store_true",
        help="ablation_loss_2",
    )

    parser.add_argument(
        "--wise_merge",
        default=False,
        action="store_true",
        help="Whether or not to use wise_merge (training)",
    )
    parser.add_argument(
        "--wise_ft",
        default=False,
        action="store_true",
        help="Whether or not to use wise_ft (evaluation)",
    )
    parser.add_argument(
        "--wise_ft_model",
        type=str,
        default="n",
        choices=["n", "zeroshot"],
        help="wise_ft_model to use.",
    )
    parser.add_argument("--wise_ft_alpha", type=float, default=0.8, help="wise_ft_alpha")

    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default="results.jsonl",
        help="Where to store the results, else does not store",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )

    # Model freeze
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning.",
    )
    parser.add_argument(
        "--freeze-fc",
        type=int,
        default=0,
        help="Whether or not to freeze the fully connection layers. Only relevant for fine-tuning.",
    )

    # evaluation setting

    # parser.add_argument("--m_class", type=int, default=10, help="Number of classes.")
    # parser.add_argument("--fc_class", type=int, default=10, help="Number of classes.")
    # parser.add_argument("--encoder_class", type=int, default=10, help="Number of classes.")

    # distance metric
    # parser.add_argument(
    #     "--distance_wiseft_fc",
    #     type=int,
    #     default=0,
    #     help="display distance between ft and zs before wiseft",
    # )
    # parser.add_argument(
    #     "--distance_ft_fc",
    #     type=int,
    #     default=0,
    #     help="display distance between ft and zs after wiseft",
    # )
    # parser.add_argument(
    #     "--feature_distance",
    #     type=int,
    #     default=0,
    #     help="display distance between features after encoder",
    # )

    # weight regularization
    # parser.add_argument(
    #     "--weight_reg",
    #     type=int,
    #     default=0,
    #     help="display distance between features after encoder",
    # )

    # lwf
    parser.add_argument("--lwf", action="store_true", help="Whether to use LWF.")
    # parser.add_argument(
    #     "--basic_model_load",
    #     type=lambda x: x.splitload(","),
    #     default=None,
    #     help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    # )
    parser.add_argument(
        "--basic_model_load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--fc_load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--keep_old_heads",
        type=int,
        default=0,
        help="display distance between features after encoder",
    )

    # BASELINE
    parser.add_argument(
        "--baseline", action="store_true", help="Whether to use BASELINE."
    )

    # trio
    parser.add_argument("--trio", action="store_true", help="Whether to use TRIO.")
    parser.add_argument(
        "--control-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on (against)",
    )
    parser.add_argument(
        "--control-dataset-add",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on (against)",
    )
    parser.add_argument(
        "noise", action="store_true", help="Whether to use random noise to regularize."
    )
    parser.add_argument("--rff", action="store_true", help="Whether to use TRIO.")

    # wise-ft
    parser.add_argument(
        "--wise-ft", action="store_true", help="Whether or not to use wise-ft"
    )
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )

    # Plusleft Fish

    parser.add_argument("--ffn_num", type=int, default=64, help="the dim of lora")
    parser.add_argument("--ffn_adapt", action="store_true", help="Whether or not to use adapter")  # use adapter
    parser.add_argument("--ffn_option", type=str, default="parallel")
    parser.add_argument("--ffn_adapt_where", type=str,
        default="AdapterDoubleEncoder",
        choices=["AdapterImageEncoder", "AdapterDoubleEncoder"],
        help="Adapter is added into ImageEncoder or ImageEncoder_and_TextEncoder.",)  # not use
    parser.add_argument("--apply_moe", action="store_true", help="Whether or not to use moe")  # use moe
    parser.add_argument("--repeat_train", action="store_true", help="default is manual differentiation")
    parser.add_argument("--task_id", type=int, default=-1, help="task_id")
    parser.add_argument("--multi_experts", action="store_true", help="Whether or not to use multi_experts")
    parser.add_argument("--experts_num", type=int, default=2, help="the number of experts")
    parser.add_argument("--is_train", action="store_true", help="Whether or not to use router noise")
    parser.add_argument("--frozen", action="store_true", help="Whether or not to use adapter")
    parser.add_argument("--autorouter", action="store_true", help="Whether or not to use autorouter")
    parser.add_argument("--task_num", type=int, default=11, help="the max_num of task, the number can be dynamically changed")
    parser.add_argument("--threshold", type=float, help="threshold for zero-shot.")
    parser.add_argument("--non_text", action="store_true", help="不使用text——encoder测试")
    parser.add_argument("--frozen-path", type=str, default="frozen_list")
    parser.add_argument("--few_shot", type=int, default=-1, help="few_shot_num")  # n-shot
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--train_chooser", action="store_true", help="train autochoose.")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (
        args.epochs is None or args.iterations is None
    ), "Cannot specify both epoch and iterations."
    assert (
        args.eval_interval is None or not args.eval_every_epoch
    ), "Cannot specify both eval_interval and eval_every_epoch."

    return args
