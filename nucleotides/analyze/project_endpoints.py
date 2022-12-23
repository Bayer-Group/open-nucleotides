import logging

import pandas as pd
import umap
from torch.utils.data import DataLoader

from nucleotides import config
from nucleotides.model.sampler import NucleotidesDataset
from nucleotides.util.util import get_target_names


def project():
    target_names = get_target_names()

    logging.info("Loading data.")
    dataloader = DataLoader(
        NucleotidesDataset(
            sequence_length=1000,
            batch_size=100000,
        ),
        batch_size=None,
        num_workers=16,
    )

    _, measured_peaks = next(iter(dataloader))
    logging.info("Running umap.")
    projector = umap.UMAP(metric="jaccard")
    projection = projector.fit_transform(measured_peaks.transpose(0, 1))
    dataframe = pd.DataFrame(projection, index=target_names)
    logging.info(f"Save to file %s", config.DISTINCT_FEATURES_UMAP)
    dataframe.to_csv(config.DISTINCT_FEATURES_UMAP, sep="\t")


if __name__ == "__main__":
    project()
