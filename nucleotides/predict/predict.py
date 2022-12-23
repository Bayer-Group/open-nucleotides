import logging
from copy import deepcopy
from typing import List

import pandas as pd
import torch
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

from nucleotides import config
from nucleotides.predict.dataloader import VCFDataSet
from nucleotides.predict.accumulator import (
    ResultAccumulator,
    Timer,
    get_number_transformed_rows,
)
from nucleotides.util.sequence_from_genome import get_nucleotide_sequence
from nucleotides.util.util import chunk_sequence, onehot_sequence


class Predictor:
    """
    Predictor class that can use the model to do predictions. Different functions
    accept either genomic coordinates or sequences to do predictions on. This
    predictor also handles sequences that are longer than the input size of the
    model by scanning over the sequence with a window function and combining the
    results.
    """

    def __init__(
            self, model, device=None, dataparallel=False, n_mc_dropout=None, batch_size=32
    ):
        self.model = model
        self.device = device or model.device
        self.batch_size = batch_size
        self.n_mc_dropout = n_mc_dropout
        if n_mc_dropout == 1:
            self.n_mc_dropout = None
        self._prepare_model(dataparallel=dataparallel)

    def _prepare_model(self, dataparallel=False):
        # Sets requires_grad to False and sets to eval mode.
        self.model.freeze()
        # For monte carlo dropout, dropout layers are set to train mode
        if self.n_mc_dropout and self.n_mc_dropout > 1:
            for module in self.model.modules():
                if module.__class__.__name__.startswith("Dropout"):
                    module.train()
        if dataparallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

    def transform_vcf(
            self,
            vcf_file,
            reference_genome=config.GRCh38,
            save_path=None,
            save_interval=None,
            continue_transform=False,
    ):

        start_index = 0
        if continue_transform:
            start_index = get_number_transformed_rows(save_path)
            if start_index:
                logging.info(
                    "Continuing %s predictions on %s after last variant saved in {}",
                    vcf_file,
                    save_path
                )

        dataset = VCFDataSet(
            vcf_file,
            reference_genome,
            sequence_length=2000,
            flip_ref_alt=False,
            start_index=start_index,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )
        self.predict_variant_batches(
            dataloader, save_path=save_path, save_interval=save_interval
        )

    def predict_variant_batches(self, dataloader, save_path=None, save_interval=None):

        with ResultAccumulator(
                save_path=save_path,
                save_interval=save_interval,
                headers={
                    "ref_predicition": self.model.hparams.target_names,
                    "alt_predicition": self.model.hparams.target_names,
                    "ref_std": self.model.hparams.target_names,
                    "alt_std": self.model.hparams.target_names,
                },
                min_itemsize={"chrom": 6, "name": 1000, "alt": 1000, "ref": 1000},
        ) as accumulator:
            timer = Timer(n_average=100, batch_size=self.batch_size)

            for batch in dataloader:
                ref = batch["encoded_ref_sequence"]
                alt = batch["encoded_alt_sequence"]
                effect = self.predict_variant_effect(ref, alt)
                del batch["encoded_ref_sequence"]
                del batch["encoded_alt_sequence"]
                effect["variant"] = deepcopy(batch)
                accumulator.add_batch(effect)
                timer()

    def predict_variant_effect(self, ref, alt):
        effect = {}
        ref_logit, effect["ref_predicition"], effect["ref_std"] = self._predict(
            ref.to(self.device)
        )
        alt_logit, effect["alt_predicition"], effect["alt_std"] = self._predict(
            alt.to(self.device)
        )
        if ref_logit.dim() > 2 and ref_logit.size(0) > 1:
            _, pvalue = ttest_ind(ref_logit, alt_logit, axis=0)
            effect["pvalue"] = pvalue
        else:
            effect["pvalue"] = None
        effect["diff"] = effect["alt_predicition"] - effect["ref_predicition"]
        return effect

    def _predict(self, encoded_sequence):
        if self.n_mc_dropout:
            logit = torch.stack(
                [self.model(encoded_sequence) for _ in range(self.n_mc_dropout)]
            )
            prediction = torch.sigmoid(logit)
            std = prediction.std(dim=0)
            prediction = prediction.mean(dim=0)
        else:
            logit = self.model(encoded_sequence)
            prediction = torch.sigmoid(logit)
            std = None
        return logit, prediction, std

    def predict_from_encoded_sequence(self, encoded_sequence, as_series=True):
        if as_series:
            return self.predict_from_encoded_sequences(
                encoded_sequence.unsqueeze(0), as_dataframe=as_series
            ).iloc[0]
        return self.predict_from_encoded_sequences(
            encoded_sequence.unsqueeze(0)
        ).squeeze()

    def predict_from_encoded_sequences(self, encoded_sequences, as_dataframe=True):
        dataloader = DataLoader(
            encoded_sequences, batch_size=self.batch_size, drop_last=False
        )
        predictions = []
        for batch in dataloader:
            _, prediction, _ = self._predict(batch.to(self.device))
            predictions.append(prediction.detach().cpu())
        predictions = torch.cat(predictions)
        if as_dataframe:
            return pd.DataFrame(predictions, columns=self.model.hparams.target_names)
        return predictions

    def predict_from_sequence(self, sequence: str, as_series=True):
        if as_series:
            return self.predict_from_sequences([sequence], as_dataframe=True).iloc[0]
        return self.predict_from_sequences([sequence], as_dataframe=False).squeeze()

    def predict_from_sequences(self, sequences: List[str], as_dataframe=True):
        chunked_sequences = []
        idx = [0]
        for sequence in sequences:
            chunked = chunk_sequence(sequence, length=self.model.hparams.sequence_length, delta=20)
            chunked_sequences.extend(chunked)
            idx.append(idx[-1] + len(chunked))
        encoded_sequences = [
            onehot_sequence(sequence) for sequence in chunked_sequences
        ]
        predictions = self.predict_from_encoded_sequences(
            encoded_sequences, as_dataframe=False
        )
        # aggregating over chunks that belong to one longer sequence
        predictions = torch.stack(
            [
                predictions[idx[i]: idx[i + 1], :].max(dim=0).values
                for i in range(len(idx) - 1)
            ]
        )
        if as_dataframe:
            return pd.DataFrame(predictions, columns=self.model.hparams.target_names)
        return predictions

    def predict_from_position(
            self, chrom, start, end, strand, reference_genome, as_series=True
    ):
        position = {
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "reference_genome": reference_genome,
        }
        if as_series:
            return self.predict_from_positions([position], as_dataframe=True).iloc[0]
        return self.predict_from_positions([position], as_dataframe=False).squeeze()

    def predict_from_positions(self, positions: List[dict], as_dataframe=True):
        sequences = [get_nucleotide_sequence(**position) for position in positions]
        return self.predict_from_sequences(sequences, as_dataframe=as_dataframe)
