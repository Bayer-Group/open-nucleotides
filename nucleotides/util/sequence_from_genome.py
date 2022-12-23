import fsspec
from selene_sdk.sequences import Genome

from nucleotides import config


def get_nucleotide_sequence(
    chrom,
    start,
    end=None,
    strand="+",
    sequence_length=1000,
    reference_genome="GRCh38",
) -> str:
    reference_sequence_path = {"GRCh37": config.GRCh37, "GRCh38": config.GRCh38}[
        reference_genome
    ]
    reference_sequence = Genome(fsspec.open(reference_sequence_path))

    if not end:
        end = start + sequence_length
    sequence = reference_sequence.get_sequence_from_coords(
        chrom, start, end, strand=strand
    )

    return sequence.upper()


if __name__ == "__main__":
    get_nucleotide_sequence(chrom="chr10", start=52750486, reference_genome="GRCh37")
