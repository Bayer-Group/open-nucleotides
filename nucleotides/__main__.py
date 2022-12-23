from typing import List, Optional

import typer
from click import Context
from rich import print
from typer.core import TyperGroup

from nucleotides import config


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    name="nucleotides",
    epilog="Thanks!",
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="""
Neural Networks Trained on Encode Data.

There are command for different stages of training and using models.
Type any sub command with --help to get guidance on the how to use
the command.

The most natural order to use the different commands is how they
are listed here. 

""",
)


@cli.command("version")
def show_version():
    """
    Show the version of the nucleotides library.
    """

    from nucleotides._version import version

    print(version)


@cli.command("settings")
def settings(setting: Optional[str] = typer.Argument("all"), verbose: bool = True):
    """Show settings. These are mostly paths to files.
    If no setting is specified, all settings are shown.

    You can overwrite specific settings
    either by changing nucleotides/settings.py or overwriting a setting using
    an environment variable prefixed with "NUCLEOTIDES".

    For instance, to overwrite the location of the reference genome
    assembly GRCh38, you can do before any command:

    NUCLEOTIDES_GRCh38=/some/custom/path nucleotides settings

    or

    NUCLEOTIDES_GRCh38=/some/custom/path nucleotides train

    You can also put these variables in a .env file and this will
    be read automatically and has precedence over anything in settings.py.
    For more information about how this works: https://pydantic-docs.helpmanual.io/usage/settings/
    """

    if setting != "all":
        key = None
        value = None
        if hasattr(config, setting):
            key = setting
            value = getattr(config, setting)
        elif hasattr(config, setting.lower()):
            key = setting.lower()
            value = getattr(config, setting.lower())
        elif hasattr(config, setting.upper()):
            key = setting.upper()
            value = getattr(config, setting.upper())
        if value:
            text = f"{key} = {value}"
            if verbose:
                text += f"""
            
To change this, either:

1) Set an environment value manually before running a command:
    
    {key}={value} nucleotides [COMMAND]

2) Add a line to a .env file:

    {key}={value}
    
3) Change the value in nucleotides/settings.py 

For more information about settings management read:
https://pydantic-docs.helpmanual.io/usage/settings/
            
            
            """
        else:
            text = f"There is no setting {setting}."

    else:
        text = str(config)

        if verbose:
            text += """
            
To change some of these settings (for example NUCLEOTIDES_GRCh38) either:

1) Set an environment value manually before running a command:
    
    NUCLEOTIDES_GRCh38=/some/custom/path nucleotides [COMMAND]

2) Add a line to a .env file:

    NUCLEOTIDES_GRCh38=/some/custom/path
    
3) Change the value in nucleotides/settings.py 

For more information about settings management read:
https://pydantic-docs.helpmanual.io/usage/settings/            
            
"""

    print(text)


@cli.command("pre-process")
def preprocess(max_files: Optional[int] = None, n_workers: int = 1):
    """Download a selection of data from ENCODE (https://www.encodeproject.org/)
    and preprocess it, so it can be used to train neural networks.
    """
    from nucleotides.resources.preprocess_all import preprocess_all

    preprocess_all(max_files=max_files, n_workers=n_workers)


@cli.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def train(context: typer.Context):
    """Train a model using Pytorch Lightning."""
    from nucleotides.model.train import train as start_train

    start_train(context.args)


@cli.command("clean-experiments")
def clean_logs():
    """Remove all pytorch lightning logs without checkpoints. These often get created while
    debugging code and clutter tensorboard.
    """
    from nucleotides.resources.clean_logs import clean

    clean()


@cli.command("show-hyperparameters", no_args_is_help=True)
def show_hyperparameters(
        experiments: List[int],
        hyperparameters: List[str] = (
                "loss_type",
                "focal_alpha",
                "focal_gamma",
                "logit_reg_neg_scale",
                "logit_reg_init_bias",
                "class_balance_beta",
                "class_balance_mode",
                "map_alpha",
                "map_beta",
                "map_gamma",
                "gpus",
                "batch_size",
                "learning_rate",
        ),
):
    """Show which hyperparamers were used for which experiment.
    Handy when tensorboard in not doing this well.
    """
    from nucleotides.util.show_hyperparameters import show

    print(show(experiments=experiments, hyperparameters=hyperparameters))


@cli.command("post-process")
def post_process(gpu: Optional[int] = None):
    """Run this command after training is done. It creates a
    few files that are necessary for downstream analysis."""

    import torch

    from nucleotides.analyze.mean_predictions import create_prediction_stats
    from nucleotides.analyze.project_endpoints import project

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)
    create_prediction_stats(device=device)
    project()


@cli.command("upload-model")
def upload_model_files_for_deployment(
        destination=config.DEPLOY_PATH, overwrite: bool = False, verbose: bool = True
):
    """This uploads a selection of files necessary for the
    API deploment. This means the model weights and other
    metadata files. This does not include code. The default
    destination for this is the setting [bold red]DEPLOY_PATH[/].
    Type "nucleotides settings --help" for more info about this.

    """
    from upath import UPath

    from nucleotides.resources.upload_to_deployed import upload

    destination = UPath(destination)

    upload(destination=destination, overwrite=overwrite, verbose=verbose)
    if verbose:
        print("Done!")


@cli.command("serve")
def serve(
        host: str = "localhost",
        port: int = 8893,
        log_level: str = "info",
        workers: int = 10,
        debug: bool = True,
):
    """Serve REST API of the model."""
    import uvicorn

    uvicorn.run(
        "nucleotides.api.main:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
        debug=debug,
    )


@cli.command("start-over")
def start_over(delete: bool = False, overwrite_archive: bool = False):
    """This will move (or delete when wanted) a set of files that that
    should be generated again when starting to train a network on
    new ENCODE data. Run this command before doing another round of
    "nucleotides preprocess".
    """
    from nucleotides.resources.start_over import clean

    clean(delete=delete, overwrite_archive=overwrite_archive)


if __name__ == "__main__":
    cli()
