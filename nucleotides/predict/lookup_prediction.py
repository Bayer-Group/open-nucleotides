import pandas as pd


def lookup_effect_predictions_in_region(predictions_hdf, chrom, start, end):
    '''
    Look up predictions in an interval on chrom from start to end.
    Args:
        predictions_hdf:
        chrom (str): chromosome, "chr3"
        start (int): start position
        end (int): end position (including)

    Returns: tuple of 3 dataframes. 1st: variant information.
             2nd: predictions reference allele.
             3rd: predictions alternative allele.

    '''
    query = "chrom=={} & pos>={} & pos<={}".format(chrom, start, end)
    variants = pd.read_hdf(predictions_hdf, key="variant", where=query)
    if len(variants) == 0:
        return [pd.DataFrame()] * 3
    query = f"index>={variants.index[0]} & index<={variants.index[-1]}"
    ref_prediction = pd.read_hdf(predictions_hdf, key="ref_prediction", where=query)
    alt_prediction = pd.read_hdf(predictions_hdf, key="alt_prediction", where=query)
    return variants, ref_prediction, alt_prediction
