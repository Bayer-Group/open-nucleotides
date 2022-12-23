from upath import UPath

from nucleotides import config


def upload(destination: UPath = config.DEPLOY_PATH, overwrite=False, verbose=True):
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

    api_dependency = inference_dependency  # + [config.PREDICTIONS]

    for path in api_dependency:
        copy_path(
            src_root=config.DATA_DIR.parent,
            dest_root=destination,
            current=path,
            overwrite=overwrite,
            verbose=verbose,
        )


def copy_path(
        src_root: UPath,
        dest_root: UPath,
        current: UPath = None,
        overwrite=False,
        verbose=True,
):
    if current is None:
        current = src_root
    if current.is_dir():
        for child in current.iterdir():
            copy_path(
                src_root=src_root,
                dest_root=dest_root,
                current=child,
                overwrite=overwrite,
            )
    if current.is_file():
        relative = current.relative_to(src_root)
        if not overwrite and relative.exists():
            print(
                f"{relative} already exists. If you want to overwrite, add --overwrite True to copy anyway"
            )
        else:
            dest = dest_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"copying {dest} to {current}...")
            dest.write_bytes(current.read_bytes())
