import logging
import time
import traceback
from collections import abc, defaultdict

import pandas as pd
import torch


class ResultAccumulator:
    """A sink for data. Handy when doing predictions on batches.
    Add the batches to the ResultAccumulator with add_batch.
    Once every save_interval batches get aggregated and saved
    to hdf5. It is saved in "table" format so that is easy
    later on to query that file and only get predictions
    in specific intervals out.
    """

    def __init__(
            self, save_path=None, save_interval=None, headers=None, min_itemsize=None
    ):
        self.save_interval = save_interval
        self.batch_counter = 0
        self._data = defaultdict(list)
        self.store = None
        self.headers = headers or {}
        self.min_itemsize = min_itemsize or {}
        if save_path:
            self.store = pd.HDFStore(save_path)

    def add_batch(self, batch_dictionary):
        for key, value in batch_dictionary.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self._data[key].append(value)
        self.batch_counter += 1
        self._save_chunk()

    def aggregate(self):
        return collate(self._data)

    def _save_chunk(self):
        if (
                not self.save_interval
                or self.store is None
                or self.batch_counter < self.save_interval
        ):
            return
        self.batch_counter = 0
        self.save(append=True)
        self._data = defaultdict(list)

    def save(self, append=False):
        data = self.aggregate()
        if isinstance(data, abc.Mapping):
            for name, value in data.items():
                nrows = 0
                if name in self.store:
                    nrows = self.store.get_storer(name).nrows
                columns = self.headers.get(name)
                dataframe = pd.DataFrame(value, columns=columns)
                dataframe.index = dataframe.index + nrows
                columns = dataframe.columns
                min_itemsize = {
                    column: self.min_itemsize[column]
                    for column in columns
                    if column in self.min_itemsize
                }
                dataframe.to_hdf(
                    self.store,
                    key=name,
                    format="t",
                    mode="a",
                    append=append,
                    min_itemsize=min_itemsize,
                )
        else:
            dataframe = pd.DataFrame(data)
            dataframe.to_hdf(self.store, key="main", format="t", mode="a", append=append)

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        self.close()


class Timer:
    def __init__(self, n_average=100, batch_size=None):
        self._time = None
        self.count = 0
        self.n_average = n_average
        self.batch_size = batch_size

    def __call__(self):
        if not self._time:
            self._time = time.time()
            return
        self.count += 1
        if self.count == self.n_average:
            now = time.time()
            delta = now - self._time
            if self.batch_size:
                per_sample = delta / (self.n_average * self.batch_size)
                logging.info("Time per 1000 samples: %s s", per_sample * 1000)
            else:
                logging.info("Time %s batches: %s s", self.n_average, delta)
            self.count = 0
            self._time = time.time()


def get_number_transformed_rows(save_path):
    nrows = 0
    if save_path:
        store = pd.HDFStore(save_path)
        if "variant" in store:
            nrows = store.get_storer("variant").nrows
        store.close()
    return nrows


def collate(data):
    if isinstance(data, str):
        return data
    elif isinstance(data, abc.Sequence):
        if isinstance(data[0], torch.Tensor):
            return torch.cat(data).numpy()
        elif isinstance(data[0], abc.Mapping):
            return {
                key: collate([point[key] for point in data]) for key in data[0].keys()
            }
        elif isinstance(data[0], list):
            return [collate(point) for batch in data for point in batch]
    elif isinstance(data, abc.Mapping):
        return {key: collate(value) for key, value in data.items()}
    else:
        return data
