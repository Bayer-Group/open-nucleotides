import pandas as pd
import torch

from nucleotides import config
from nucleotides.optimize_sequence.genetic_torch import GeneticAlgorithm
from nucleotides.predict.inference_model import get_model


def optimize_interactive(
        model,
        starting_population=None,
        positive_features=None,
        negative_features=None,
        other_features="mask",
        n_generations=2000,
        sequence_length=None,
        population_size=1024,
        mutation_rate=0.005,
        n_tournaments=3,
        auto_balance_loss=True,
        penalize_distance=True,
):
    device = model.device

    target, mask = create_target(
        all_features=model.hparams.feature_names,
        positive_features=positive_features,
        negative_features=negative_features,
        other_features=other_features,
    )

    algorithm = GeneticAlgorithm(
        model,
        starting_population=starting_population,
        target=target,
        mask=mask,
        sequence_length=sequence_length,
        population_size=population_size,
        mutation_rate=mutation_rate,
        n_tournaments=n_tournaments,
        auto_balance_loss=auto_balance_loss,
        penalize_distance=penalize_distance,
        device=device,
    )

    algorithm.fit(n_generations=n_generations)
    return algorithm


def create_target(
        all_features,
        positive_features=None,
        negative_features=None,
        other_features="mask",
):
    if "other_features" not in ("mask", "positive", "negative"):
        raise ValueError(
            "Argument 'other_features' should have value of 'mask', 'positive', 'negative'"
        )
    if other_features == "positive":
        target = torch.ones(len(all_features))
    else:
        target = torch.zeros(len(all_features))
    if positive_features:
        positive_indices = [all_features.index(feat) for feat in positive_features]
        target[positive_indices] = 1
    if negative_features:
        negative_indices = [all_features.index(feat) for feat in negative_features]
        target[negative_indices] = 0
    if other_features == "mask":
        mask = torch.zeros(len(all_features))
        mask[positive_indices] = 1
        mask[negative_indices] = 1
    else:
        mask = torch.ones(len(all_features))
    return target, mask


if __name__ == "__main__":
    optimize()
