import logging
import os
import subprocess
from multiprocessing import Pool

import pandas as pd

from nucleotides import config
from nucleotides.resources.process_peaks import process_narrowPeak
from nucleotides.resources.select_experiments import (
    filter_chip_metadata,
    filter_dnase_metadata,
)
from nucleotides.resources.select_training_intervals import select_training_intervals
from nucleotides.resources.util import download_if_necessary


def preprocess_all(
    meta_dnase_url=config.META_DNASE_URL,
    meta_chip_url=config.META_CHIP_URL,
    meta_dnase_file=config.META_DNASE_FILE,
    meta_chip_file=config.META_CHIP_FILE,
    out_file=config.PREPROCESSED,
    endpoints_file=config.DISTINCT_FEATURES,
    selected_experiment_file=config.SELECTED_EXPERIMENTS,
    max_files=None,
    n_workers=1,
):
    encode_peaks, encode_metadata = preprocess_encode(
        meta_dnase_url=meta_dnase_url,
        meta_chip_url=meta_chip_url,
        meta_dnase_file=meta_dnase_file,
        meta_chip_file=meta_chip_file,
        max_files=max_files,
        n_workers=n_workers,
    )

    manual_peaks, manual_metadata = get_manually_added_files()

    labeled_peaks = pd.concat([encode_peaks, manual_peaks]).sort_values([0, 1, 2])
    metadata = pd.concat([encode_metadata, manual_metadata])

    # Write dataframes to disk
    labeled_peaks.to_csv(out_file, sep="\t", index=False, header=False)
    metadata.to_csv(selected_experiment_file, sep="\t")

    training_intervals = select_training_intervals(
        intervals=labeled_peaks, selected_experiments=metadata
    )
    training_intervals.to_csv(config.INTERVALS, sep="\t", index=False, header=False)
    bgzip_and_tabix(out_file)

    # Get distinct features (ids) actually present
    logging.info("Writing names of endpoints to file.")
    distinct_features = pd.DataFrame(labeled_peaks[3].unique())
    distinct_features.to_csv(endpoints_file, sep="\t", index=False, header=False)

    # Download the reference genome
    logging.info("Downloading reference genome.")
    reference_genome = get_reference_genome(url=config.GRCh38_URL, file=config.GRCh38)
    logging.info("Done Preprocessing.")


def preprocess_encode(
    meta_dnase_url=config.META_DNASE_URL,
    meta_chip_url=config.META_CHIP_URL,
    meta_dnase_file=config.META_DNASE_FILE,
    meta_chip_file=config.META_CHIP_FILE,
    max_files=None,
    n_workers=1,
):
    # Get experiment metadata from the ENCODE website
    metadata_dnase = get_metadata(url=meta_dnase_url, path=meta_dnase_file)
    metadata_chip = get_metadata(url=meta_chip_url, path=meta_chip_file)

    metadata_chip, _ = filter_chip_metadata(metadata_chip)
    metadata_dnase, _ = filter_dnase_metadata(metadata_dnase)

    selected_experiments = pd.concat([metadata_dnase, metadata_chip]).iloc[:max_files]

    logging.info("Number of selected experiments: {}".format(len(selected_experiments)))

    # Adding a factor column with a nicely
    # formatted name of the Transcription factor DNAse
    selected_experiments["factor"] = (
        selected_experiments["Experiment target"]
        .fillna("DNAse")
        .map(lambda x: x.split("-")[0])
    )

    selected_experiments["label"] = None
    selected_experiments["label_sensitive"] = None
    selected_experiments["processed_file"] = None
    selected_experiments["processed_file_sensitive"] = None

    selected_narrowpeak = selected_experiments[
        selected_experiments["File format type"] == "narrowPeak"
    ]

    # Download the necessary bed files
    bed_files = get_bed_files(urls=selected_narrowpeak["File download URL"][:max_files])
    selected_narrowpeak["bedfile"] = bed_files

    # Create groups that should be merged into one prediction task
    replicate_groups = [
        (group, bioname, factor)
        for (bioname, factor), group in selected_narrowpeak.groupby(
            ["Biosample term name", "factor"]
        )
    ]

    logging.info(
        "Number of replicate groups resulting in prediction task: {}".format(
            len(replicate_groups)
        )
    )

    pool = Pool(processes=n_workers)
    labeled_peaks, selected_narrowpeak = zip(
        *pool.starmap(merge_replicates_narrowpeak, replicate_groups)
    )
    selected_narrowpeak = pd.concat(selected_narrowpeak)

    logging.info("Concatenating data from all tissues and targets and sorting.")
    labeled_peaks = pd.concat(labeled_peaks)

    return labeled_peaks, selected_narrowpeak


def get_manually_added_files():
    labeled_peaks = []
    metadata = []
    for file_path in config.BED_MANUALLY_ADDED_DIR.iterdir():
        logging.info(f"Adding manually added file {file_path}")
        label = file_path.stem.split(".")[0]
        labeled_peaks.append(get_labeled_dataframe(file_path, label))
        metadata.append([label, file_path, "DNase"])
    labeled_peaks = pd.concat(labeled_peaks)
    return labeled_peaks, pd.DataFrame(metadata, columns=["label", "bedfile", "factor"])


def merge_replicates_narrowpeak(metadata, bioname, factor):
    files = process_narrowPeak(
        metadata["bedfile"], bioname, factor, out_dir=config.BED_PREPROCESSED_DIR
    )
    # metadata.loc[metadata.index, ["processed_file", "processed_file_sensitive"]] = [
    #     files
    # ] * len(metadata)

    # metadata.loc[metadata.index, ["label", "label_sensitive"]] = [
    #     [f.stem.split(".")[0] for f in files]
    # ] * len(metadata)

    # Adding the resulting processed files to metadata file
    metadata.loc[metadata.index, "processed_file"] = files * len(metadata)
    metadata.loc[metadata.index, "label"] = [f.stem.split(".")[0] for f in files] * len(
        metadata
    )

    labeled_data = pd.concat(
        [
            get_labeled_dataframe(bed_file, bed_file.stem.split(".")[0])
            for bed_file in files
        ]
    )
    return labeled_data, metadata


def get_metadata(url, path=None):
    """Download (ENCODE) metadata file into a pandas dataframe."""
    if path and path.exists():
        return pd.read_csv(path, sep="\t", index_col=0)
    logging.info("Downloading: {}".format(url))
    df = pd.read_csv(url, sep="\t")
    if path:
        df.to_csv(path, sep="\t")
    return df


def get_bed_files(urls):
    """Download the bed files (if not downloaded yet) and save them in the
    bed file directory (default: resources/training_data/bed_files).
    Subsequently resources them (taking only the first 3 columns and adding
    a fourth one with the file_accession to create a label). Concatenate
    all into essentially one large bed and sort by the first 3 columns (genome positions).
    """
    logging.info(
        "Collecting individual bed files, downloading only if not present yet."
    )
    bed_files = [
        download_if_necessary(url, directory=config.BED_DOWNLOAD_DIR) for url in urls
    ]
    return bed_files


def get_labeled_dataframe(path, label):
    """Get a bed file in the desired shape selecting the first 3 columns and
    adding a label as the fourth column.
    """

    dataframe = pd.read_csv(path, sep="\t", compression="gzip", header=None)
    dataframe = dataframe[[0, 1, 2]]
    dataframe[3] = label
    return dataframe


def bgzip_and_tabix(bed_file):
    """Run bgzip and tabix on a bed_file."""
    logging.info("Running bgzip.")
    bgzipped = bed_file.with_suffix(".bed.gz")
    with open(bgzipped, "w") as outfile:
        subprocess.run(["bgzip", "-f", bed_file, "--stdout"], stdout=outfile)
    logging.info("Running tabix")
    subprocess.run(["tabix", "-f", "-p", "bed", bgzipped])
    return bgzipped


def get_reference_genome(url=config.GRCh38_URL, file=config.GRCh38):
    reference_genome = download_if_necessary(
        url, file=config.GRCh38.with_suffix(".fasta.gz")
    )
    if reference_genome.exists():
        logging.info("Decompressing reference genome: {}".format(reference_genome))
        subprocess.run(["bgzip", "-d", reference_genome])
    return file


def gunzip(path):
    if not path.suffix == ".gz":
        return path
    if path.exists() and not (path.parent / path.stem).exists():
        subprocess.run(["gunzip", path])
    return path.parent / path.stem


def add_arguments(parser):
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit the number of bed files that are downloaded and processed. Handy for debugging.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of processes")
    return parser


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        filename=config.PREPROCESSING_LOG,
    )
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    pd.set_option("display.max_columns", 500)
    args = parser.parse_args()
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.workers)
    preprocess_all(max_files=args.max_files, n_workers=args.workers)
