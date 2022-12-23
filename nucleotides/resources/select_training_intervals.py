"""
We could take all intervals in preprocessed.bed to train on and let the sampler sample
n basepair training sequences with the middle basepair somewhere within those intervals.

However, the vast majority is DNAse. To speed up training a bit for the other endpoints
we can try to get a bit of a healthier mix. In the DeepSea paper they are sampling only
from regions where there is at least one TF binding. Of course there would be also
most likely be DNAse data for those same regions and therefore DNAse endpoints get trained
as well.

Here I will mix. Get all intervals from not-DNAse and mix an equal amount of DNAse intervals
in there.

"""

import logging

import pandas as pd


def select_training_intervals(
        intervals: pd.DataFrame,
        selected_experiments: pd.DataFrame,
        n_samples=None,
        at_least_one_non_dnase=True,
):
    logging.info("Sampling training intervals...")

    dnase_experiments = selected_experiments[selected_experiments["factor"] == "DNase"]
    other_experiments = selected_experiments[selected_experiments["factor"] != "DNase"]

    intervals_dnase_experiments = intervals[
        intervals[3].isin(dnase_experiments["label"])
    ]
    intervals_other_experiments = intervals[
        intervals[3].isin(other_experiments["label"])
    ]

    if at_least_one_non_dnase:
        if n_samples:
            sampled_intervals = intervals_other_experiments.sample(n=n_samples)
        else:
            sampled_intervals = intervals_other_experiments

    else:
        if n_samples:
            intervals_dnase_experiments = intervals_dnase_experiments.sample(
                n=n_samples
            )
            intervals_other_experiments = intervals_other_experiments.sample(
                n=n_samples
            )
        else:
            intervals_dnase_experiments = intervals_dnase_experiments.sample(
                n=len(intervals_other_experiments)
            )

        sampled_intervals = pd.concat(
            [intervals_dnase_experiments, intervals_other_experiments], axis=0
        )

    logging.info(f"Number of sampled training intervals: {len(sampled_intervals)}")

    sampled_intervals = merge_overlapping_intervals(sampled_intervals)
    return sampled_intervals


def merge_overlapping_intervals(intervals: pd.DataFrame):
    """
    Merge rows with overlapping intervals.
    """
    logging.info(f"Merging overlapping intervals, number of intervals before merge: {len(intervals)}")
    intervals = pd.concat(
        [
            merge_intervals_one_chromosome(chrom_df)
            for _, chrom_df in intervals.groupby(0)
        ]
    )
    logging.info(f"Number of intervals after merge: {len(intervals)}")
    return intervals


def merge_intervals_one_chromosome(chrom_df):
    chrom_df = chrom_df.sort_values(by=1)
    chrom_df["group"] = ((chrom_df[1] > chrom_df[2].shift().cummax())).cumsum()
    result = chrom_df.groupby("group").agg({0: "first", 1: "min", 2: "max"})
    return result
