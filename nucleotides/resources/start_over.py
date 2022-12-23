import datetime
import pathlib
import shutil

from nucleotides import config

files_to_clean = [
    config.META_DNASE_FILE,
    config.META_CHIP_FILE,
    config.SELECTED_EXPERIMENTS,
    config.PREPROCESSED,
    config.DISTINCT_FEATURES,
    config.TARGET,
    config.TARGET.with_suffix(".gz.tbi"),
    config.INTERVALS,
    config.POSITIVE_WEIGHTS,
    config.PREPROCESSING_LOG,
    config.BED_PREPROCESSED_DIR,
    config.DISTINCT_FEATURES_UMAP,
    config.MEAN_PREDICTIONS,
]


def clean(delete: bool = False, overwrite_archive: bool = False):
    if delete:
        delete_files()
    else:
        archive_files(overwrite_archive=overwrite_archive)


def delete_files():
    """
    Removing files to start with a clean slate.
    """
    for path in files_to_clean:
        print(f"Removing {path}.")
        remove(path)


def archive_files(overwrite_archive=False):
    archive_dir = create_archive_path()
    if archive_dir.exists() and overwrite_archive is False:
        print(
            f"{archive_dir} already exists. Use --overwrite-archive if you want to overwrite."
        )
        return
    archive_dir.mkdir(exist_ok=True)
    print(f"Archiving old files to {archive_dir}")
    for path in files_to_clean:
        if not path.exists():
            continue
        archive_path = archive_dir / path.relative_to(config.DATA_DIR)
        if archive_path.exists() and overwrite_archive is False:
            print(
                f"Skipping moving {path} to {archive_path} because {archive_path} already exists. Use --overwrite if you want to overwrite."
            )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Moving {path} to {archive_path}")
        shutil.move(path, archive_dir / archive_path)


def create_archive_path():
    return config.DATA_DIR / datetime.datetime.now().strftime("archive_%d_%m_%Y")


def remove(path: pathlib.Path):
    if path.is_file():
        remove_file(path)
    elif path.is_dir():
        empty_entire_dir(path)


def remove_file(path: pathlib.Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def empty_entire_dir(path: pathlib.Path):
    for child in path.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            empty_entire_dir(child)
            child.rmdir()
