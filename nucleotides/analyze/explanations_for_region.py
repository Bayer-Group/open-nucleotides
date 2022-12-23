import numpy as np
import pandas as pd

from nucleotides.analyze.explain_by_permutation import explain
from nucleotides.util.sequence_from_genome import get_nucleotide_sequence
from nucleotides.util.util import chunk_region


def explain_region(
        predictor,
        chrom,
        start,
        end,
        strand="+",
        reference_genome="GRCh38",
        length=1000,
        delta=20,
):
    sequence = get_nucleotide_sequence(
        chrom=chrom,
        start=start,
        end=end,
        strand=strand,
        reference_genome=reference_genome,
    )
    explainations = explain_sequence(
        predictor, sequence=sequence, length=length, delta=delta
    )
    explainations.columns = list(range(start, end))
    return explainations


def explain_sequence(predictor, sequence, length=1000, delta=20):
    """Get explainations for sequence longer than input
    of neural net.
    """

    end = len(sequence)
    start_idx = chunk_region(0, end, length=length, delta=delta)
    # chunks = [sequence[start_id:start_id + length] for start_id in start_idx]

    target_names = predictor.model.hparams.target_names
    n_endpoints = len(target_names)

    explanation_matrix_width = start_idx[-1] + length
    empty_2d_array = np.full([n_endpoints, explanation_matrix_width], np.nan)

    expanations_subsequences = []
    for start_id in start_idx:
        sequence_chunk = sequence[start_id: start_id + length]
        explanation = explain(predictor, sequence_chunk, organize=False).to_numpy()
        offset = start_id
        print(
            f"calculate explanation subsequence with start_id {start_id} and offset {offset}"
        )
        matrix = empty_2d_array.copy()
        matrix[:, offset: offset + length] = explanation
        expanations_subsequences.append(matrix)
    expanations_subsequences = np.stack(expanations_subsequences, axis=2)
    explanation_aggregated = np.nanmean(expanations_subsequences, axis=2)
    df = pd.DataFrame(
        explanation_aggregated,
        columns=list(range(start_idx[-1] + length)),
        index=target_names,
    )
    return df
