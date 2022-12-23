"""
An script that will process (merge, intersect, etc) a set of peaks depending on the type of resources that it is.
The basic rule of the directory structure should be (e.g.):

Processed data ends up in data/bed_files/processed.

"""

import logging
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from nucleotides import config

narrowPeak_action_dict = {
    "DNase": [
        {"merge": True, "intersect": True, "slop": 0, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 0, "output": "_sensitive.bed"},
    ],
    "H3K27ac": [
        {"merge": True, "intersect": True, "slop": 0, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 0, "output": "_sensitive.bed"},
    ],
    "H3K27me3": [
        {"merge": True, "intersect": True, "slop": 300, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 300, "output": "_sensitive.bed"},
    ],
    "H3K36me3": [
        {"merge": True, "intersect": True, "slop": 200, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 200, "output": "_sensitive.bed"},
    ],
    "H3K4me1": [
        {"merge": True, "intersect": True, "slop": 100, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 100, "output": "_sensitive.bed"},
    ],
    "H3K4me3": [
        {"merge": True, "intersect": True, "slop": 0, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 0, "output": "_sensitive.bed"},
    ],
    "H3K9ac": [
        {"merge": True, "intersect": True, "slop": 0, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 0, "output": "_sensitive.bed"},
    ],
    "H3K9me3": [
        {"merge": True, "intersect": True, "slop": 0, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 0, "output": "_sensitive.bed"},
    ],
    "other": [
        {"merge": True, "intersect": True, "slop": 0, "output": ".bed"},
        # {"merge": True, "intersect": False, "slop": 0, "output": "_sensitive.bed"},
    ],
}
broadPeak_action_dict = {
    "other": [
        {"merge": True, "intersect": False, "slop": 0, "output": "_sensitive.bed.gz"},
        {
            "merge": False,
            "intersect": True,
            "slop": 0,
            "output": "_high_confidence.bed.gz",
        },
    ]
}


def process_narrowPeak(bed_files, bioname, factorName, out_dir):
    logging.info("Processing bed files for {} and {}".format(bioname, factorName))
    out_dir = Path(out_dir) / "narrowPeak"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=str(config.TEMP_DIR) + "/"))
    bed_files = [sort_bed_file(bed_file, temp_dir) for bed_file in bed_files]
    out_files = process_factor(
        bed_files,
        bioname,
        factorName,
        out_dir,
        temp_dir,
        action_dict=narrowPeak_action_dict,
    )
    clean_and_remove_directory(temp_dir)
    return out_files


def process_factor(
        bed_files,
        bio_name,
        factor_name,
        out_dir,
        temp_dir,
        action_dict=None,
):
    if narrowPeak_action_dict is None:
        action_dict = narrowPeak_action_dict
    out_files = []
    for actions in action_dict.get(factor_name, action_dict["other"]):
        bio_name = "_".join(bio_name.split())
        out_file = "{}_{}{}".format(bio_name, factor_name, actions["output"])
        out_file = out_dir / re.sub("[^\\.\w\s-]", "", out_file).strip().lower()

        slop_dir = temp_dir / "processed"
        slop_size = actions["slop"]
        working_files = [
            do_slop(bed_file, slop_size, slop_dir) for bed_file in bed_files
        ]

        if len(working_files) == 1:
            shutil.move(working_files[0], out_file)
        elif actions["merge"] and actions["intersect"]:
            do_merge_and_intersect(working_files, out_file)
        elif actions["merge"]:
            do_merge(working_files, out_file)
        elif actions["intersect"]:
            do_intersect(working_files, out_file)
        out_file = bgzip(out_file)
        out_files.append(out_file)
    return out_files


def process_peak(f_in, f_out, min_score):
    dataframe = pd.read_csv(BytesIO(f_in), sep="\t", index_col=0, header=None)
    dataframe["peak"] = ["peak_{}".format(i + 1) for i in range(len(dataframe))]
    dataframe = dataframe[dataframe[3] >= min_score]
    dataframe[3] = dataframe[3].fillna(".")
    dataframe = dataframe[[1, 2, "peak", 3]]
    dataframe.to_csv(f_out, sep="\t", header=False)


def sort_bed_file(path, out_dir):
    out_file = (out_dir / path.stem).with_suffix(".bed")
    dataframe = pd.read_csv(path, sep="\t", header=None)
    dataframe = dataframe.sort_values([0, 1, 2])
    dataframe.to_csv(out_file, sep="\t", index=False, header=False)
    # if path.suffix == '.bed':
    #    cmd = 'cat ' + str(path) + ' | sort -k 1,1 -k 2,2n -k 3,3n'
    #    with open(out_file, "w") as f:
    #        subprocess.call(cmd, shell=True, stdout=f)
    # elif path.suffix == '.gz':
    #    cmd = 'zcat ' + str(path) + ' | sort -k 1,1 -k 2,2n -k 3,3n'
    #    with open(str(out_file), "w") as f:
    #        subprocess.call(cmd, shell=True, stdout=f)

    return out_file


def do_slop(in_file, slop_size, slop_dir, genome_size_file=config.HG38_SIZES):
    slop_dir.mkdir(parents=True, exist_ok=True)
    out_file = slop_dir / in_file.name
    if not slop_size:
        shutil.copy(in_file, slop_dir)
        return out_file
    # cmd = (
    #         "bedtools slop -b "
    #         + str(slop_size)
    #         + " -i "
    #         + str(in_file)
    #         + " -g "
    #         + str(genome_size_file)
    # )
    cmd = [
        "bedtools",
        "slop" "-b",
        str(slop_size),
        "-i",
        str(in_file),
        "-g",
        str(genome_size_file),
    ]
    with open(out_file, "w") as open_file:
        subprocess.call(cmd, shell=False, stdout=open_file)
    return out_file


def do_merge_and_intersect(working_file_list, out_file):
    working_file_list = [str(path) for path in working_file_list]
    intersect_temp = out_file.with_suffix(".temp")

    # minCov = max(len(working_file_list) - 1, 2)  # currently ignored in favour of a 1

    # cmd = "multiIntersectBed -cluster -i " + " ".join(workingFileList)
    cmd = ["multiIntersectBed", "-cluster", "-i"] + [
        str(path) for path in working_file_list
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=False)
    out, _ = p.communicate()
    with open(intersect_temp, "w") as f:
        process_peak(out, f, min_score=1)

    ##And then we merge with our results
    cmd = (
            "cat "
            + " ".join(working_file_list)
            + " | sort -k 1,1 -k 2,2n -k 3,3n -S 50% | bedtools intersect -u -a stdin -b "
            + str(intersect_temp)
            + " | sort -k 1,1 -k 2,2n -k 3,3n -S 50% | bedtools merge"
    )
    with open(out_file, "w") as f:
        # Need shell=True for the pipes. Input is not dynamically defined by user.
        subprocess.call(cmd, shell=True, stdout=f)  # nosec
    intersect_temp.unlink()
    return out_file


def do_merge(workingFileList, out_file):
    workingFileList = [str(path) for path in workingFileList]
    cmd = (
            "cat "
            + " ".join(workingFileList)
            + " | sort -k 1,1 -k 2,2n -k 3,3n -S 50% | bedtools merge"
    )
    with open(out_file, "w") as f:
        # Need shell=True for the pipes. Input is not dynamically defined by user.
        subprocess.call(cmd, shell=True, stdout=f)  # nosec
    return out_file


def do_intersect(workingFileList, out_file):
    workingFileList = [str(path) for path in workingFileList]
    min_cov = max(len(workingFileList) - 1, 2)

    # cmd = "multiIntersectBed -cluster -i " + " ".join(workingFileList)
    cmd = ["multiIntersectBed", "-cluster", "-i"] + [
        str(path) for path in workingFileList
    ]
    buffer = BytesIO()
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
    out, _ = process.communicate()
    process_peak(out, buffer, min_score=min_cov)
    # cmd = "bedtools merge"
    cmd = ["bedtools", "merge"]
    with open(out_file, "w") as open_file:
        subprocess.call(cmd, shell=False, stdin=buffer, stdout=open_file)
    return out_file


def bgzip(bed_file):
    if bed_file.suffix == ".gz":
        return bed_file
    bgzipped = bed_file.with_suffix(".bed.gz")
    with open(bgzipped, "w") as outfile:
        subprocess.run(["bgzip", "-f", bed_file, "--stdout"], stdout=outfile)
    bed_file.unlink()
    return bgzipped


def clean_and_remove_directory(path):
    shutil.rmtree(path)
