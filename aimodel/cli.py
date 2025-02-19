from pathlib import Path
from typing import get_args

import click
import numpy as np
from rich import print
from rich.table import Table
from torch.cuda import device_count

from aimodel.data.loader import FolderType, load_corpus, save_sentences
from aimodel.passive import is_passive_sentence
from aimodel.sentence_splitter import split_sentence
from aimodel.server import run_app
from aimodel.trainer import Config, PassiveClassifier, TrainConfig


@click.group()
def cli() -> None:
    """
    The main entry point to this project.
    """


@cli.command()
def run_server() -> None:
    """
    Run the FastApi server that serves the passive sentence classifier.
    """
    run_app()


@cli.command()
@click.option(
    "--folder",
    "-f",
    type=click.Choice(get_args(FolderType)),
    required=True,
    help="The folder to compute passive sentences from.",
    prompt=True,
)
def compute_passive(folder: FolderType) -> None:
    """
    Split a raw text file to passive and non-passive sentences.
    Then store it back to the same folder.

    Parameters
    ----------
    folder : FolderType
        The folder from which to load the raw text file.
    """
    passive: list[str] = []
    non_passive: list[str] = []

    for line in load_corpus(folder):
        for sentence in split_sentence(line):
            if is_passive_sentence(sentence):
                passive.append(sentence)
            else:
                non_passive.append(sentence)

    save_sentences(folder, passive, non_passive)


@cli.command()
@click.option(
    "--folder",
    "-f",
    type=click.Choice(get_args(FolderType)),
    required=True,
    help="The folder to compute stats from.",
    prompt=True,
)
def compute_passive_stats(folder: FolderType) -> None:
    """
    Compute the stats of the passive sentences in the given folder.
    Like mean, std, of sentence length etc.

    Parameters
    ----------
    folder : FolderType
        The folder from which to load the data.
    """
    passive_corpus = load_corpus(folder, "passive")
    non_passive_corpus = load_corpus(folder, "non_passive")
    passive_lengths = [len(sentence) for sentence in passive_corpus]
    non_passive_lengths = [len(sentence) for sentence in non_passive_corpus]

    table = Table(title="Passive Sentences infos")
    table.add_column("Metric", justify="center")
    table.add_column("Value", justify="center", style="cyan")
    table.add_row("# Passive Sentences", str(len(passive_corpus)))
    table.add_row("# Non Passive Sentences", str(len(non_passive_corpus)))
    table.add_row("Mean Length of Passive Sentences", str(np.mean(passive_lengths)))
    table.add_row(
        "Mean Length of Non Passive Sentences",
        str(np.mean(non_passive_lengths)),
    )
    table.add_row("Std Length of Passive Sentences", str(np.std(passive_lengths)))
    table.add_row(
        "Std Length of Non Passive Sentences",
        str(np.std(non_passive_lengths)),
    )
    table.add_row("Min Length of Passive Sentences", str(np.min(passive_lengths)))
    table.add_row(
        "Min Length of Non Passive Sentences",
        str(np.min(non_passive_lengths)),
    )
    table.add_row("Max Length of Passive Sentences", str(np.max(passive_lengths)))
    table.add_row(
        "Max Length of Non Passive Sentences",
        str(np.max(non_passive_lengths)),
    )

    print(table)


# TODO make an attrs to click options converter (I need to finish it)
@cli.command()
@click.option(
    "--backbone",
    "-b",
    type=str,
    required=True,
    help="The huggingface backbone model name.",
    default="bert-base-uncased",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["-1"] + [str(gpu_idx) for gpu_idx in range(device_count())]),
    required=True,
    help="The device (gpu number or -1 for cpu) on which to train.",
)
@click.option(
    "--checkpoint",
    "-c",
    required=False,
    default=None,
    help="The weight and biases or S3 bucket checkpoint to load.",
)
@click.option(
    "--entity",
    "-e",
    required=False,
    default=None,
    help="The weight and biases entity to use.",
)
@click.option(
    "--project",
    "-p",
    required=False,
    default=None,
    help="The weight and biases project to use.",
)
def train_model(
    backbone: str,
    device: str,
    checkpoint: str | None,
    entity: str,
    project: str,
) -> None:
    """
    Train a passive sentence classifier.

    Parameters
    ----------
    backbone : str
        The huggingface model to finetune.
    device : str
        The device to use (-1 for CPU else the GPU idx)
    checkpoint : Optional[str]
        The previously trained checkpoint to restart from.
    """
    device_idx = int(device)

    model = PassiveClassifier(
        config=Config(
            backbone=backbone,
            entity=entity,
            project=project,
            cuda_device=device_idx,
        ),
    )

    train_config = TrainConfig(
        train_folder=Path(__file__)
        .parent.joinpath("data/bush_gore")
        .expanduser()
        .resolve(),
        valid_folder=Path(__file__)
        .parent.joinpath("data/bush_kerry")
        .expanduser()
        .resolve(),
        learning_rate=5e-3,
        checkpoint=checkpoint,
        learning_rate_decay=0.90,
        epochs=30,
        warmup_steps=100,
        gradient_accumulation_steps=4,
        logging_steps=50,
        valid_batch_size=2,
        train_batch_size=2,
        precision=32,
        validation_check_every=1,
    )

    model.launch_train(train_config=train_config)


@cli.command()
@click.option("--checkpoint", "-c", required=True, help="The checkpoint to load.")
def optimize_model(checkpoint: str) -> None:
    """
    Optimize a previously trained model to ONNX.

    Parameters
    ----------
    checkpoint : str
        The checkpoint to load and optimize.
    """
    PassiveClassifier.optimize(checkpoint)
