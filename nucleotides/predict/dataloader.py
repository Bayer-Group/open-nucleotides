import logging

from selene_sdk.predict._common import get_reverse_complement_encoding
from selene_sdk.predict._variant_effect_prediction import (
    _handle_long_ref,
    _handle_standard_ref,
    _process_alt,
    read_vcf_file,
)
from selene_sdk.sequences import Genome
from torch.utils.data import Dataset

from nucleotides import config


class VCFDataSet(Dataset):
    """Pytorch Dataset for Variant Call Format (VCF), wrapping
       functionality provided by Selene.
    """

    def __init__(
            self,
            vcf_path,
            reference_sequence_path=config.GRCh38,
            sequence_length=1000,
            flip_ref_alt=False,
            start_index=0,
            pad=True,
    ):
        super().__init__()
        logging.info("Creating Pytorch Dataset")
        self.sequence_length = sequence_length
        self.reference_sequence_path = reference_sequence_path
        logging.info("Reading VCF file")
        self.variants = read_vcf_file(
            vcf_path,
            reference_sequence=Genome(str(reference_sequence_path)),  # stringify_path
        )
        self.variants = self.variants[start_index:]
        logging.info("Done creating PyTorch Dataset")
        logging.info(
            "First variant in the dataset is {} {} {} {} {} {}".format(
                *self.variants[0]
            )
        )
        self.flip_ref_alt = flip_ref_alt
        self.pad = pad

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, index):
        variant = self.variants[index]
        chrom, pos, name, ref, alt, strand = variant
        if self.flip_ref_alt:
            ref, alt = alt, ref
        (
            ref_sequence_encoding,
            alt_sequence_encoding,
            contains_unknown,
            match,
        ) = get_encoded_sequences(
            chrom,
            pos,
            name,
            ref,
            alt,
            strand,
            self.sequence_length,
            self.pad,
            self.reference_sequence_path,
        )

        return {
            "encoded_ref_sequence": ref_sequence_encoding,
            "encoded_alt_sequence": alt_sequence_encoding,
            "chrom": chrom,
            "pos": pos,
            "name": name,
            "ref": ref,
            "alt": alt,
            "strand": strand,
            "contains_unknown_bases": contains_unknown,
            "matches_reference_genome": match,
        }


def get_encoded_sequences(
        chrom,
        pos,
        name,
        ref,
        alt,
        strand="+",
        sequence_length=1000,
        pad=True,
        reference_sequence_path=config.GRCh38,
):
    reference_sequence = Genome(
        reference_sequence_path
    )  # Has to happen here because pyaidx is not process save

    start_radius = end_radius = sequence_length // 2
    if sequence_length % 2:
        start_radius += 1
    center = pos + len(ref) // 2
    start = center - start_radius
    end = center + end_radius
    (
        ref_sequence_encoding,
        contains_unknown,
    ) = reference_sequence.get_encoding_from_coords_check_unk(
        chrom, start, end, pad=pad
    )
    ref_encoding = reference_sequence.sequence_to_encoding(ref)
    alt_sequence_encoding = _process_alt(
        chrom, pos, ref, alt, start, end, ref_sequence_encoding, reference_sequence
    )

    match = True
    seq_at_ref = None
    if len(ref) and len(ref) < sequence_length:
        match, ref_sequence_encoding, seq_at_ref = _handle_standard_ref(
            ref_encoding, ref_sequence_encoding, sequence_length, reference_sequence
        )
    elif len(ref) >= sequence_length:
        match, ref_sequence_encoding, seq_at_ref = _handle_long_ref(
            ref_encoding,
            ref_sequence_encoding,
            start_radius,
            end_radius,
            reference_sequence,
        )

    if contains_unknown:
        logging.warning(
            "For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
            "reference sequence contains unknown base(s)"
            "--will be marked `True` in the `contains_unknown` column "
            "of the .tsv or the row_labels .txt file.".format(
                chrom, pos, name, ref, alt, strand
            )
        )
    if not match:
        logging.warning(
            "For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
            "reference does not match the reference genome. "
            "Reference genome contains {6} instead. "
            "Predictions/scores associated with this "
            "variant--where we use '{3}' in the input "
            "sequence--will be marked `False` in the `ref_match` "
            "column of the .tsv or the row_labels .txt file".format(
                chrom, pos, name, ref, alt, strand, seq_at_ref
            )
        )
    if strand == "-":
        ref_sequence_encoding = get_reverse_complement_encoding(
            ref_sequence_encoding,
            reference_sequence.BASES_ARR,
            reference_sequence.COMPLEMENTARY_BASE_DICT,
        )
        alt_sequence_encoding = get_reverse_complement_encoding(
            alt_sequence_encoding,
            reference_sequence.BASES_ARR,
            reference_sequence.COMPLEMENTARY_BASE_DICT,
        )

    return (
        ref_sequence_encoding.transpose(),
        alt_sequence_encoding.transpose(),
        contains_unknown,
        match,
    )
