import pathlib

from pydantic import BaseConfig, BaseSettings
from upath import UPath


class EnvironmentSettings(BaseSettings):
    environment: str = "develop"

    class Config(BaseConfig):
        env_prefix = "nucleotides_"
        env_file = ".env"

    def __str__(self):
        return super().__str__().replace(" ", "\n")


class Settings(BaseSettings):
    _environment_settings: EnvironmentSettings = EnvironmentSettings()

    ENCODE_BASE_URL: str = "https://www.encodeproject.org/"

    META_DNASE_URL: str = (
            ENCODE_BASE_URL
            + "metadata/?type=Experiment&assay_title=DNase-seq&status=released&assembly=GRCh38&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=ATAC-seq&files.file_type=bigWig&files.file_type=bed+narrowPeak&files.file_type=bigBed+narrowPeak"
    )
    META_CHIP_URL: str = (
            ENCODE_BASE_URL
            + "metadata/?type=Experiment&assay_title=TF+ChIP-seq&assay_title=Histone+ChIP-seq&status=released&assembly=GRCh38&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&files.file_type=bigWig&files.file_type=bigBed+narrowPeak&files.file_type=bed+narrowPeak&files.file_type=bed+broadPeak&files.file_type=bigBed+broadPeak"
    )

    GRCh38_URL: str = (
            ENCODE_BASE_URL
            + "files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
    )

    # Set this variable to a path to some bucket/place that
    # can be seen by the deployed process
    DEPLOY_PATH: UPath = None

    if _environment_settings.environment == "production":
        TOP_DIR: UPath = DEPLOY_PATH
        INFERENCE_MODEL_CLASS = None
        INFERENCE_MODEL: UPath = None
    else:
        TOP_DIR: UPath = UPath(pathlib.Path.home() / "data/nucleotides")
        INFERENCE_MODEL_CLASS = None
        INFERENCE_MODEL: UPath = None

    DATA_DIR: UPath = TOP_DIR / "data"
    TEMP_DIR: UPath = DATA_DIR / "temp_dir"
    LIGHTNING_LOGS: UPath = TOP_DIR / "lightning_logs"

    PREPROCESSING_LOG: UPath = DATA_DIR / "preprocessing.log"

    # TISSUE_SAMPLES = DATA_DIR / "tissue_samples.txt"
    HG38_SIZES: UPath = DATA_DIR / "hg38.sizes.txt"

    # Data Used for Training the models
    TRAINING_DIR: UPath = DATA_DIR / "training_data"

    META_DNASE_FILE: UPath = TRAINING_DIR / "metadata_dnase_data_narrow_bed.tsv"
    META_CHIP_FILE: UPath = TRAINING_DIR / "metadata_all_chip.tsv"

    # The following files will be generated when running the prepocess_all.py script and
    # will be subsequenty used in the model scripts:

    # Subset of the metadata only containing file accession we want to use for model:
    SELECTED_EXPERIMENTS: UPath = TRAINING_DIR / "selected_experiments.tsv"

    # File containing a list of the unique chromatin features cellline combinations
    # (i.e. the output nodes of the neural network). What we are actually listing
    # are the file accession ids.
    DISTINCT_FEATURES: UPath = TRAINING_DIR / "distinct_features.tsv"

    # The preprocessed and concatenated form of the files
    # bed directory ends up here:
    PREPROCESSED: UPath = TRAINING_DIR / "preprocessed.bed"
    # TARGET: UPath = PREPROCESSED.with_suffix(".bed.gz")
    TARGET: UPath = TRAINING_DIR / "preprocessed.bed.gz"

    INTERVALS: UPath = TRAINING_DIR / "intervals.bed"

    # The reference genome will be stored here:
    GENOME_DIR: UPath = DATA_DIR / "genome"
    GRCh37: UPath = GENOME_DIR / "hg19.fa"
    GRCh38: UPath = GENOME_DIR / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"

    # bed files from encode are downloaded to this directory:
    BED_DIR: UPath = TRAINING_DIR / "bed_files"
    BED_DOWNLOAD_DIR: UPath = BED_DIR / "download"
    BED_MANUALLY_ADDED_DIR: UPath = BED_DIR / "manually_added"
    BED_PREPROCESSED_DIR: UPath = BED_DIR / "preprocessed"

    POSITIVE_WEIGHTS: UPath = TRAINING_DIR / "positive_weights.csv"

    PREDICTIONS: UPath = DATA_DIR / "predictions_repaired.hdf5"

    SEQUENCE_OPTIMIZATION: UPath = DATA_DIR / "sequence_optimization"

    DISTINCT_FEATURES_UMAP: UPath = DATA_DIR / "umap_distinct_features.tsv"
    MEAN_PREDICTIONS: UPath = DATA_DIR / "mean_predictions.tsv"

    # Only used in some places, like determining which device to
    # use to serve the API.
    DEVICE: str = "cpu"

    class Config(BaseConfig):
        env_prefix = "nucleotides_"
        env_file = ".env"

    def __str__(self):
        return (
                str(self._environment_settings)
                + "\n"
                + super().__str__().replace(" ", "\n")
        )


if __name__ == "__main__":
    settings = Settings()
    print(settings.DATA_DIR)
    print(settings.BED_MANUALLY_ADDED_DIR)
