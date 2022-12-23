"""
Running predictions on randomly selected sequences to get
statistics about the background predictions. i.e. what is the
mean and standard deviation for predictions of the different
endpoints.
"""

import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from nucleotides import config
from nucleotides.model.sampler import NucleotidesDataset
from nucleotides.predict.inference_model import get_model


def create_prediction_stats(device="cpu"):
    model = get_model().to(device)
    target_names = model.hparams.target_names

    logging.info("Loading data.")
    dataloader = DataLoader(
        NucleotidesDataset(
            sequence_length=1000,
            batch_size=512,
        ),
        batch_size=None,
        num_workers=16,
    )

    predictions = []
    for i, (features, target) in enumerate(dataloader):
        predictions.append(
            pd.DataFrame(
                torch.sigmoid(model(features.to(device).transpose(1, 2).float()))
                .detach()
                .cpu()
                .numpy(),
                columns=target_names,
            )
        )
        if i > 1000:
            break
    predictions = pd.concat(predictions).transpose()
    print(predictions)

    df = pd.DataFrame(
        {
            "mean prediction": predictions.mean(axis=1),
            "std prediction": predictions.std(axis=1),
        }
    )
    print(df)
    logging.info(f"Save to file %s", config.MEAN_PREDICTIONS)
    df.to_csv(config.MEAN_PREDICTIONS, sep="\t")
