import numbers
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nucleotides import config
from nucleotides.loss.loss import add_args as add_loss_args
from nucleotides.loss.loss import get_loss
from nucleotides.metrics.average_precision import MultitaskBinnedAveragePrecision
from nucleotides.metrics.confusion_matrix import MultitaskMetrics, pr_scatter_plot
from nucleotides.model.sampler import NucleotidesDataset, get_positive_weight


class FunctionalModel(pl.LightningModule):
    """Base Class for Pytorch Lightning Models.
    """

    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)

        self.train_metrics = MultitaskMetrics(
            num_tasks=hparams.n_targets, task_names=hparams.target_names, prefix="train"
        )
        self.val_metrics = MultitaskMetrics(
            num_tasks=hparams.n_targets, task_names=hparams.target_names, prefix="val"
        )
        self.test_metrics = MultitaskMetrics(
            num_tasks=hparams.n_targets, task_names=hparams.target_names, prefix="test"
        )

        self.train_pr = MultitaskBinnedAveragePrecision(
            task_names=hparams.target_names, prefix="train"
        )
        self.val_pr = MultitaskBinnedAveragePrecision(
            task_names=hparams.target_names, prefix="val"
        )
        self.test_pr = MultitaskBinnedAveragePrecision(
            task_names=hparams.target_names, prefix="test"
        )

        if isinstance(self.hparams.positive_weight, numbers.Number):
            self.hparams.positive_weight = (
                    torch.ones(self.hparams.n_targets) * self.hparams.positive_weight
            )

        positive_weight = hparams.positive_weight
        if hparams.max_positive_weight:
            positive_weight = positive_weight.clamp_max(hparams.max_positive_weight)

        self.loss_function = get_loss(hparams)
        self.learning_rate = self.hparams.learning_rate

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.transpose(1, 2).float())
        loss = self.loss_function(logits, y.float())

        pred = torch.sigmoid(logits)
        output = self.train_metrics(pred, y.bool()) or {}
        self.log_dict(output)
        output = self.train_pr(pred, y.bool()) or {}
        self.log_dict(output)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            NucleotidesDataset(
                sequence_length=self.hparams.sequence_length,
                batch_size=self.hparams.batch_size,
            ),
            batch_size=None,
            num_workers=16,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.transpose(1, 2).float())
        loss = self.loss_function(logits, y.float())

        pred = torch.sigmoid(logits)
        output = self.val_metrics(pred, y.bool()) or {}
        self.log_dict(output)
        output = self.val_pr(pred, y.bool()) or {}
        self.log_dict(output)

        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        precision_recall_aucs = self.val_pr.compute_for_tasks()["val_pr_auc"].to_numpy()

        self.logger.experiment.add_histogram(
            "precision_recall_aucs", precision_recall_aucs, global_step=self.global_step
        )
        self.logger.experiment.add_figure(
            "precision_vs_recall",
            self.val_metrics.plot(),
            global_step=self.global_step,
        )

        try:
            per_endpoint_logdir = Path(self.logger.log_dir) / "per_endpoint_metrics"
            per_endpoint_logdir.mkdir(exist_ok=True, parents=True)
            dataframe = pd.concat(
                [self.val_pr.compute_for_tasks(), self.val_metrics.compute_for_tasks()],
                axis=1,
            )
            dataframe.to_csv(per_endpoint_logdir / f"val_{self.global_step}.tsv", sep="\t")
        except AttributeError:
            pass

    def val_dataloader(self):
        return DataLoader(
            NucleotidesDataset(
                mode="validate",
                sequence_length=self.hparams.sequence_length,
                batch_size=self.hparams.batch_size,
            ),
            batch_size=None,
            num_workers=8,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.transpose(1, 2).float())
        loss = self.loss_function(logits, y.float())

        pred = torch.sigmoid(logits)
        output = self.test_metrics(pred, y.bool()) or {}
        self.log_dict(output)
        output = self.test_pr(pred, y.bool()) or {}
        self.log_dict(output)

        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        precision_recall_aucs = self.test_pr.compute_for_tasks()[
            "test_pr_auc"
        ].to_numpy()
        stats = self.val_metrics.compute_for_tasks()
        precisions = stats["test_precision"].to_numpy()
        recalls = stats["test_recall"].to_numpy()
        self.logger.experiment.add_histogram(
            "precision_recall_aucs", precision_recall_aucs, global_step=self.global_step
        )
        self.logger.experiment.add_figure(
            "precision_vs_recall",
            pr_scatter_plot(precisions, recalls),
            global_step=self.global_step,
        )

        per_endpoint_logdir = Path(self.logger.log_dir) / "per_endpoint_metrics"
        per_endpoint_logdir.mkdir(exist_ok=True, parents=True)
        df = pd.concat(
            [self.val_pr.compute_for_tasks(), self.val_metrics.compute_for_tasks()],
            axis=1,
        )
        df.to_csv(per_endpoint_logdir / f"val_{self.global_step}.tsv", sep="\t")

    def test_dataloader(self):
        return DataLoader(
            NucleotidesDataset(
                mode="test",
                sequence_length=self.hparams.sequence_length,
                batch_size=self.hparams.batch_size,
            ),
            batch_size=None,
            num_workers=8,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--sequence_length", type=int, default=1000)
        parser.add_argument(
            "--target_names",
            type=list,
            default=config.DISTINCT_FEATURES.read_text().splitlines(),
        )
        parser.add_argument(
            "--n_targets",
            type=int,
            default=len(config.DISTINCT_FEATURES.read_text().splitlines()),
        )
        parser.add_argument(
            "--positive_weight", type=float, default=get_positive_weight()
        )

        parser.add_argument("--max_positive_weight", type=float, default=None)
        parser.add_argument("--batch_size", type=int, default=64)
        parser = add_loss_args(parser)
        return parser
