import logging
import os
from pathlib import Path

import pytest

from nucleotides import config

basic_paths = [
    config.DATA_DIR,
    config.GENOME_DIR,
]

preprocess_dependency = [config.HG38_SIZES]

# Generated for reference but not necessary for further tasks
generated_during_preprocess = [
    config.META_DNASE_FILE,
    config.META_CHIP_FILE,
    config.SELECTED_EXPERIMENTS,
    config.TEMP_DIR,
    config.BED_DIR,
    config.BED_DOWNLOAD_DIR,
    config.BED_PREPROCESSED_DIR,
    config.PREPROCESSED,
]

train_paths_dependency = [
    config.DISTINCT_FEATURES,
    config.TARGET,
    config.INTERVALS,
    config.POSITIVE_WEIGHTS,
    config.GRCh38,
]

inference_dependency = [
    config.DISTINCT_FEATURES,
    config.INFERENCE_MODEL,
    config.DISTINCT_FEATURES_UMAP,
    config.SELECTED_EXPERIMENTS,
    config.MEAN_PREDICTIONS,
    config.GRCh37,
    config.GRCh38,
    config.GRCh37.with_suffix(config.GRCh37.suffix + ".fai"),
    config.GRCh38.with_suffix(config.GRCh38.suffix + ".fai"),
]

api_dependency = inference_dependency + [config.PREDICTIONS]


def requires_env(*envs):
    env = os.environ.get("TESTENV")
    logging.info(env)
    envs = envs if isinstance(envs, list) else [*envs]

    return pytest.mark.skipif(
        env not in envs and env is not None,
        reason=f"Not suitable envrionment {env} for current test",
    )


def skip_in_env(*envs):
    env = os.environ.get("TESTENV")
    logging.info(env)
    envs = envs if isinstance(envs, list) else [*envs]

    return pytest.mark.skipif(
        env in envs and env is not None,
        reason=f"Not suitable envrionment {env} for current test",
    )


@pytest.mark.parametrize("path", basic_paths)
def test_basic_path_is_available(path: Path):
    assert path.exists()


@skip_in_env("prod")
@pytest.mark.parametrize("path", preprocess_dependency)
def test_preprocess_dependency_path_is_available(path: Path):
    assert path.exists()


# @skip_in_env("prod")
# @pytest.mark.parametrize("path", generated_during_preprocess)
# def test_generated_during_preprocess_path_is_available(path: Path):
#    assert path.exists()


@skip_in_env("preprocess")
@skip_in_env("prod")
@pytest.mark.parametrize("path", train_paths_dependency)
def test_train_dependency_path_is_available(path: Path):
    assert path.exists()


@skip_in_env("preprocess")
@skip_in_env("train")
@pytest.mark.parametrize("path", inference_dependency)
def test_inference_dependency_path_is_available(path: Path):
    assert path.exists()


@skip_in_env("preprocess")
@skip_in_env("train")
@pytest.mark.parametrize("path", api_dependency)
def test_api_dependency_path_is_available(path: Path):
    assert path.exists()
