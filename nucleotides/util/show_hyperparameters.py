from argparse import ArgumentParser

import pandas as pd
import yaml

from nucleotides import config


def show(experiments, hyperparameters):
    hparams_list = []
    index = []
    experiments = [f"version_{exp}" for exp in experiments] + experiments
    for experiment_dir in config.LIGHTNING_LOGS.iterdir():
        if not str(experiment_dir.stem) in experiments:
            continue
        hparam_file = experiment_dir / "hparams.yaml"
        hparams = read_yaml(hparam_file)
        hparams_list.append({key: hparams[key] for key in hyperparameters})
        index.append(experiment_dir.stem)
    return pd.DataFrame(hparams_list, index=index)


def read_yaml(path):
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown
    )
    with open(path, "r", encoding="ascii") as stream:
        # about bandit: flagging this as unsafe load
        # is unjustified, see implementation of
        # SafeLoaderIgnoreUnknown
        return yaml.load(stream, Loader=SafeLoaderIgnoreUnknown)  # nosec


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=[409, 410, 411, 412, 413, 414, 415, 416, 417, 418],
    )
    parser.add_argument(
        "-p",
        "--hyperparameters",
        nargs="+",
        default=[
            "loss_type",
            "focal_alpha",
            "focal_gamma",
            "logit_reg_neg_scale",
            "logit_reg_init_bias",
            "class_balance_beta",
            "class_balance_mode",
            "map_alpha",
            "map_beta",
            "map_gamma",
            "gpus",
            "batch_size",
            "learning_rate",
        ],
    )
    args = parser.parse_args()
    print(show(args.experiments, args.hyperparameters))
