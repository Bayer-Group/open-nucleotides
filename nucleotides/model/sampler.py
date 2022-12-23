import random

import pandas as pd
import torch
from selene_sdk.samplers import IntervalsSampler
from selene_sdk.sequences import Genome
from torch.utils.data import DataLoader, IterableDataset

from nucleotides import config
from nucleotides.util.util import get_target_names

target_names = get_target_names()


class NucleotidesDataset(IterableDataset):
    """Pytorch IterableDataset wrapping Selene's
    IntervalSampler.
    """

    def __init__(
            self,
            reference_sequence_path=config.GRCh38,
            target_path=config.TARGET,
            distinct_features=target_names,
            intervals_path=config.INTERVALS,  # intervals_path,
            sample_negative=False,
            validation_holdout=("chr6", "chr7"),
            test_holdout=("chr8", "chr9"),
            sequence_length=1000,
            center_bin_to_predict=200,
            feature_thresholds=0.5,
            mode="train",
            save_datasets=("test"),
            output_dir=None,
            batch_size=32,
    ):
        super().__init__()
        self.reference_sequence_path = reference_sequence_path
        self.target_path = target_path
        self.distinct_features = distinct_features
        self.intervals_path = intervals_path
        self.sample_negative = sample_negative
        self.validation_holdout = list(validation_holdout)
        self.test_holdout = list(test_holdout)
        self.sequence_length = sequence_length
        self.center_bin_to_predict = center_bin_to_predict
        self.feature_thresholds = feature_thresholds
        self.mode = mode
        self.save_datasets = save_datasets
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.sampler = None
        self.n = 0

    def __iter__(self):
        self._set_up_sampler()
        self.n = 0
        return self

    def __next__(self):
        if (
                self.n
                <= len(self.sampler._sample_from_mode[self.sampler.mode].indices)
                // self.batch_size
        ):
            self.n += 1
            return self.sampler.sample(self.batch_size)
        raise StopIteration

    def _set_up_sampler(self):
        self.sampler = IntervalsSampler(
            reference_sequence=Genome(
                str(self.reference_sequence_path)
            ),  # stringify_path
            target_path=str(self.target_path),  # stringify_path
            features=self.distinct_features,
            intervals_path=self.intervals_path,
            sample_negative=self.sample_negative,
            seed=random.randint(1, 9999),
            validation_holdout=self.validation_holdout,
            test_holdout=self.test_holdout,
            sequence_length=self.sequence_length,
            center_bin_to_predict=self.center_bin_to_predict,
            feature_thresholds=self.feature_thresholds,
            mode=self.mode,
            save_datasets=self.save_datasets,
            output_dir=self.output_dir,
        )


def get_positive_weight(batches=10000, from_file=True):
    if from_file and config.POSITIVE_WEIGHTS.exists():
        return torch.tensor(pd.read_csv(config.POSITIVE_WEIGHTS, header=None)[0])
    dataset = NucleotidesDataset(batch_size=32, mode="train", sequence_length=400)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=8)
    positive_frequency = torch.zeros(len(target_names), requires_grad=False)
    for i, (_, targets) in enumerate(dataloader):
        positive_frequency += targets.sum(axis=0).detach()
        batch_size = len(targets)
        positive_weights = (
                                   (i + 1) * batch_size - positive_frequency
                           ) / positive_frequency
        positive_weights[positive_weights == float("inf")] = 1
        print(positive_weights)
        if i == batches:
            break
    if from_file:
        pd.DataFrame(positive_weights).to_csv(
            config.POSITIVE_WEIGHTS, index=None, header=None
        )
    return positive_weights


if __name__ == "__main__":
    print("calculating positive weights")
    print(get_positive_weight(batches=10000))
