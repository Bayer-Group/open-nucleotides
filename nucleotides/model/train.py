import logging
import sys
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, loggers

import nucleotides
from nucleotides import config
from nucleotides.model.pinkpanther import PinkPanther
from nucleotides.model.woodpecker import Woodpecker

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

model_classes = {
    "pinkpanther": PinkPanther,
    "woodpecker": Woodpecker,
}


def train(args):
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", default=0.00005, type=float)
    parser.add_argument("--model_class", default="deepsea", type=str)

    # adding "name" here, because LIGHTNING_LOGS already points to the
    # lightning_logs directory. Otherwise the logging directory becomes
    # lightning_logs/lightning_logs/version_foo
    tb_logger = loggers.TensorBoardLogger(save_dir=config.LIGHTNING_LOGS, name="")

    parser = nucleotides.model.lightning_model.FunctionalModel.add_model_specific_args(
        parser
    )
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args(args)
    model_class = model_classes[hparams.model_class]
    trainer = Trainer.from_argparse_args(hparams, logger=tb_logger)
    model = model_class(hparams)

    # Running a dummy batch through the network
    # to instantiate lazy modules
    dummy_batch = torch.rand(2, 4, hparams.sequence_length)
    model(dummy_batch)

    trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    train(sys.argv[1:])
