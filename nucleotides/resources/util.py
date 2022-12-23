import logging
import urllib.request
from pathlib import Path


def download_if_necessary(url, directory=None, file=None):
    if directory and file:
        file_path = directory / file
    elif file:
        file_path = file
    elif directory:
        name = url.rsplit("/", 1)[-1]
        file_path = directory / name
    else:
        file_path = Path(url.rsplit("/", 1)[-1])
    # Do not download again if the file already exists or the
    # a the file exists without the extension (i.e. already unzipped for instance)
    if not file_path.exists() and not (file_path.parent / file_path.stem).exists():
        logging.info(f"Downloading: {url}")
        urllib.request.urlretrieve(url, file_path)  # nosec
    return file_path


def select(dataframe, criteria):
    for column, value in criteria:
        dataframe = dataframe[dataframe[column] == value]
    return dataframe
