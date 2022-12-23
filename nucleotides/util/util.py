from pathlib import Path

import pandas as pd
import torch

from nucleotides import config


def sequence_to_ints(sequence: str):
    return ["ACGT".index(string) for string in sequence]


def ints_to_sequence(ints: int):
    return "".join(["ACGT"[index] for index in ints])


def onehot_sequence(sequence: str):
    sequence = torch.LongTensor(sequence_to_ints(sequence))
    eye = torch.eye(4)
    return eye[sequence].transpose(0, 1)


def get_target_names():
    return config.DISTINCT_FEATURES.read_text().splitlines()


def get_endpoint_description():
    umap = get_endpoint_umap_coordinates()
    metadata = map_endpoints_to_metadata(umap.index)
    stats = map_endpoints_to_stats()
    return pd.concat([metadata, umap, stats], axis=1)


def get_endpoint_umap_coordinates():
    umap = pd.read_csv(config.DISTINCT_FEATURES_UMAP, sep="\t", index_col=0)
    umap.columns = ["umap1", "umap2"]
    umap.index.name = "endpoint"
    return umap


def map_endpoints_to_metadata(endpoints):
    metadata = pd.read_csv(config.SELECTED_EXPERIMENTS, sep="\t")
    metadata = metadata[
        ["Experiment target", "Biosample type", "Biosample term name", "label"]
    ]
    metadata.drop_duplicates(subset=["label"], keep="first", inplace=True)
    metadata.set_index("label", inplace=True)
    endpoint_data = metadata.reindex(endpoints)
    return endpoint_data


def map_endpoints_to_stats():
    stats = pd.read_csv(config.MEAN_PREDICTIONS, sep="\t", index_col=0)
    stats.index.name = "endpoint"
    return stats


def chunk_region(start, end, length=1000, delta=20, overhang=False, include_ends=True):
    middle = start + (end - start) // 2
    reference = middle - (length // 2)
    index = reference
    start_idx = [index]
    index -= delta

    while index >= start:  # - delta:
        start_idx.append(index)
        index -= delta
    if start_idx[-1] != start:
        if overhang:
            start_idx.append(index)
        elif include_ends:
            start_idx.append(start)
    start_idx.reverse()
    index = reference + delta
    while index <= end - length:  # + delta:
        start_idx.append(index)
        index += delta
    if start_idx[-1] != end - length:
        if overhang:
            start_idx.append(index)
        elif include_ends:
            start_idx.append(end - length)
    return start_idx


def chunk_sequence(sequence, length=1000, delta=20):
    if length == len(sequence):
        return [sequence]
    start_idx = chunk_region(0, len(sequence), length=length, delta=delta)
    chunks = [sequence[start_id: start_id + length] for start_id in start_idx]
    return chunks


def cache_pandas(path: Path):
    def decorator(func):
        def new_func(*args, **kwargs):
            if path.exists():
                return pd.read_csv(path, sep="\t", index_col=0)
            dataframe = func(*args, **kwargs)
            dataframe.to_csv(path, sep="\t")
            return dataframe

        return new_func

    return decorator


if __name__ == "__main__":
    print(chunk_sequence("0123456789" * 100))
