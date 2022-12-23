import shutil

from nucleotides import config


def clean():
    counter = 0
    for experiment in config.LIGHTNING_LOGS.iterdir():
        if not experiment.is_dir():
            continue
        if not (experiment / "checkpoints").exists():
            shutil.rmtree(experiment, ignore_errors=True)
            counter += 1
            continue
        checkpoints = list((experiment / "checkpoints").iterdir())
        if not checkpoints:
            shutil.rmtree(experiment, ignore_errors=True)
            counter += 1
    print(f"removed {counter} experiments without checkpoints.")


def rmdir(directory):
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


if __name__ == "__main__":
    clean()
