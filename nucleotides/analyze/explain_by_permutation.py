import pandas as pd
import torch

from nucleotides.analyze.explain import organize_explanation_matrix
from nucleotides.predict.inference_model import get_model
from nucleotides.util.util import sequence_to_ints


def create_all_point_mutations(sequence):
    bases = ["A", "C", "G", "T"]
    return [
        replace_base(sequence, i, alternative)
        for i, base in enumerate(sequence)
        for alternative in bases
    ]


def replace_base(sequence, index, new_base):
    as_list = list(sequence)
    as_list[index] = new_base
    return "".join(as_list)


def original_index(sequence):
    ints = torch.IntTensor(sequence_to_ints(sequence)) + torch.arange(
        start=0, end=4 * len(sequence), step=4
    )
    # ints = [4 * i + b for i, b in enumerate(sequence_to_ints(sequence))]
    return ints


def predict_all_point_mutations(predictor, sequence):
    sequences = create_all_point_mutations(sequence)
    predictions: torch.Tensor = predictor.predict_from_sequences(
        sequences, as_dataframe=False
    )
    return predictions


def explain(predictor, sequence, endpoints=None, organize=True):
    predictions = predict_all_point_mutations(predictor, sequence)
    wild_type_predictions = predictions[original_index(sequence)]
    predictions = predictions.reshape(
        predictions.shape[0] // 4, 4, predictions.shape[1]
    ).transpose(0, 1)
    difference = predictions - wild_type_predictions
    average_difference = difference.sum(dim=0) / 3.0
    # most_negative = difference.min(dim=0)[0].clip(max=0)
    # most_positive = difference.max(dim=0)[0].clip(min=0)
    explanations = pd.DataFrame(
        average_difference.transpose(0, 1).cpu().numpy(), index=predictor.model.hparams.target_names
    )
    if endpoints:
        explanations = explanations.loc[endpoints]
    if organize:
        explanations = organize_explanation_matrix(explanations)
    return explanations


if __name__ == "__main__":
    from nucleotides.predict.predict import Predictor

    dummy_sequence = "ATCG" * 250
    dummy_explanations = explain(Predictor(get_model()), dummy_sequence)
    print(dummy_explanations)
