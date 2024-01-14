import os
import sys
from ast import literal_eval

import configargparse

sys.path.append("src")

from dataset import DataIterativeLoader
from trainer import BaseDATrainer, UnlabeledDATrainer, get_trainer
from util import arguments_parsing, set_seed, wandb_logger

import pdb

@wandb_logger(
    keys=[
        "method",
        "source",
        "target",
        "seed",
        "num_iters",
        "alpha",
        "T",
        "update_interval",
        "lr",
        "warmup",
        "note",
    ]
)
def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    set_seed(args.seed)

    # match args.method.split("_"):
    #     case "cdac", *_:
    #         loaders = DataIterativeLoader(args, strong_transform=True)
    #     case _:
    #         loaders = DataIterativeLoader(args, strong_transform=False)

    # match args.method.split("_"):
    #     case "base", *label_trick:
    #         trainer = get_trainer(BaseDATrainer, label_trick)(loaders, args)
    #     case ("mme" | "cdac") as unlabeled_method, *label_trick:
    #         trainer = get_trainer(UnlabeledDATrainer, label_trick)(
    #             loaders, args, unlabeled_method=unlabeled_method
    #         )

    
    # if args.method.split("_") == "cdac":
    #     loaders = DataIterativeLoader(args, strong_transform=True)
    # else:
    #     loaders = DataIterativeLoader(args, strong_transform=False)

    # pdb.set_trace()

    # if args.method.split("_") == "base":
    #     trainer = get_trainer(BaseDATrainer, label_trick)(loaders, args)
    # elif args.method.split("_") == "mme":
    #         trainer = get_trainer(UnlabeledDATrainer, label_trick)(
    #             loaders, args, unlabeled_method=unlabeled_method
    #         )


    label_trick = None
    loaders = DataIterativeLoader(args, strong_transform=False)
    trainer = get_trainer(BaseDATrainer, label_trick)(loaders, args)



    trainer.train()


if __name__ == "__main__":
    args = arguments_parsing("config.yaml")
    # replace the configuration
    args.dataset = args.dataset_cfg["dataset_cfg"][args.dataset]
    main(args)
