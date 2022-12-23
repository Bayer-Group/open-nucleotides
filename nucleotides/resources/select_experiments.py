import logging

import pandas as pd


def filter_chip_metadata(metadata, tissue_file=None):
    logging.info("Selecting ChIP Experiments")

    selection = metadata[~metadata["File format"].isin(["fastq", "bam", "tagAlign"])]
    selection = selection[selection["File assembly"] == "GRCh38"]
    selection = selection[selection["File Status"] == "released"]
    selection = selection[selection["File analysis status"] == "released"]

    selection = selection[selection["Library crosslinking method"].isna()]
    logging.info(
        f"Selecting Library crosslinking method == NaN {len(selection)}"
    )

    selection = selection[selection["Biosample treatments"].isna()]
    logging.info(f"Selecting Biosample treatments == NaN {len(selection)}")

    selection = selection[selection["Biosample genetic modifications methods"].isna()]
    logging.info(
        f"Selecting Biosample genetic modifications methods == NaN {len(selection)}"
    )

    selection = selection[
        selection["Biosample genetic modifications categories"].isna()
    ]
    logging.info(
        f"Selecting Biosample genetic modifications categories == NaN {len(selection)}"
    )

    selection = selection[selection["Audit ERROR"].isna()]
    logging.info(f"Selecting Audit ERROR == NaN {len(selection)}")

    selection = selection[selection["Audit NOT_COMPLIANT"].isna()]
    logging.info(f"Selecting Audit NOT_COMPLIANT == NaN {len(selection)}")

    # selecting a subset of tissues
    if tissue_file:
        tissue_data = pd.read_csv(tissue_file, sep="\t")
        tissue_data = tissue_data[tissue_data["keep"] == "y"]
        selection = selection[
            selection["Biosample term name"].isin(tissue_data["bioname"])
        ]

    bigwig_data = select_bigwig_experiments(
        selection[selection["File type"] == "bigWig"]
    )
    bed_data = selection[
        (selection["File type"] == "bed")
        & selection["File format type"].isin(["narrowPeak", "broadPeak"])
        ]
    bed_data = select_bed_files(bed_data)

    bigwig_data = bigwig_data[
        [
            "File download URL",
            "Biological replicate(s)",
            "Experiment target",
            "Biosample type",
            "Biosample term name",
            "Project",
            "File format type",
        ]
    ]

    bigwig_data["File format type"] = "coverage"

    bed_data = bed_data[
        [
            "File download URL",
            "Biological replicate(s)",
            "Experiment target",
            "Biosample type",
            "Biosample term name",
            "Project",
            "File format type",
        ]
    ]

    logging.info(
        f"Selected {len(bed_data)} bed files and {len(bigwig_data)} bigwig files"
    )
    return bed_data, bigwig_data


def filter_dnase_metadata(metadata):
    logging.info("Selecting DNAse Experiments")
    logging.info(f"Original number of experiments: {len(metadata)}")

    selection = metadata[~metadata["File format"].isin(["fastq", "bam", "tagAlign"])]
    logging.info(
        f"Selecting file format fatstq, bam and tagAlign: {len(selection)}"
    )

    selection = selection[selection["File assembly"] == "GRCh38"]
    logging.info(f"Selecting File Assembly == GRCh38 {len(selection)}")

    selection = selection[selection["File Status"] == "released"]
    logging.info(f"Selecting File Status == released {len(selection)}")

    selection = selection[selection["File analysis status"] == "released"]
    logging.info(f"Selecting File analysis status == released {len(selection)}")

    selection["Experiment target"] = "DNase"

    selection = selection[selection["Library crosslinking method"].isna()]
    logging.info(
        f"Selecting Library crosslinking method == NaN {len(selection)}"
    )

    selection = selection[selection["Biosample treatments"].isna()]
    logging.info(f"Selecting Biosample treatments == NaN {len(selection)}")

    selection = selection[selection["Biosample genetic modifications methods"].isna()]
    logging.info(
        f"Selecting Biosample genetic modifications methods == NaN {len(selection)}")

    selection = selection[
        selection["Biosample genetic modifications categories"].isna()
    ]
    logging.info(
        f"Selecting Biosample genetic modifications categories == NaN {len(selection)}"
    )

    bigwig_data = select_bigwig_experiments(
        selection[selection["File type"] == "bigWig"]
    )
    bed_data = selection[
        (selection["File type"] == "bed")
        & selection["File format type"].isin(["narrowPeak", "broadPeak"])
        ]
    logging.info(
        f"Bed files: selecting File type == bed and File format type is narrowPeak or broadPeak {len(bed_data)}"
    )

    bed_data = select_bed_files(bed_data)
    logging.info(f"Bed files: selecting replicates {len(bed_data)}")

    bigwig_data = bigwig_data[
        [
            "File download URL",
            "Biological replicate(s)",
            "Experiment target",
            "Biosample type",
            "Biosample term name",
            "Project",
            "File format type",
        ]
    ]

    bigwig_data["File format type"] = "coverage"

    bed_data = bed_data[
        [
            "File download URL",
            "Biological replicate(s)",
            "Experiment target",
            "Biosample type",
            "Biosample term name",
            "Project",
            "File format type",
        ]
    ]

    logging.info(
        f"Selected {len(bed_data)} bed files and {len(bigwig_data)} bigwig files"
    )
    return bed_data, bigwig_data


def select_bigwig_experiments(dataframe):
    selected = []
    for _, group in dataframe.groupby(
            ["Biosample term name", "Experiment target", "Project"]
    ):
        if (group["Output type"] == "signal p-value").any():
            group = group[group["Output type"] == "signal p-value"]
        elif (group["Output type"] == "read-depth normalized signal").any():
            group = group[group["Output type"] == "read-depth normalized signal"]
        if (group["Biological replicate(s)"] == "1, 2").any():
            group = group[group["Biological replicate(s)"] == "1, 2"]
        if (group["Library crosslinking method"].isna()).any():
            group = group[group["Library crosslinking method"].isna()]
        selected.append(group)
    return pd.concat(selected)


def select_bed_files(dataframe):
    selected = []
    for _, group in dataframe.groupby(
            ["Biosample term name", "Experiment target", "Project"]
    ):
        if (group["Biological replicate(s)"] == "1, 2, 3, 4").any():
            group = group[group["Biological replicate(s)"] == "1, 2, 3, 4"]
        elif (group["Biological replicate(s)"] == "1, 2, 3").any():
            group = group[group["Biological replicate(s)"] == "1, 2, 3"]
        elif (group["Biological replicate(s)"] == "1, 2").any():
            group = group[group["Biological replicate(s)"] == "1, 2"]
        if (group["Library crosslinking method"].isna()).any():
            group = group[group["Library crosslinking method"].isna()]
        if (group["Output type"] == "optimal IDR thresholded peaks").any():
            group = group[group["Output type"] == "optimal IDR thresholded peaks"]
        elif (group["Output type"] == "replicated peaks").any():
            group = group[group["Output type"] == "replicated peaks"]
        elif (group["Output type"] == "stable peaks").any():
            group = group[group["Output type"] == "stable peaks"]
        elif (group["Output type"] == "pseudoreplicated IDR thresholded peaks").any():
            group = group[
                group["Output type"] == "pseudoreplicated IDR thresholded peaks"
                ]
        selected.append(group)
    return pd.concat(selected)
