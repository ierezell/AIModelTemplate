from pathlib import Path
from typing import Literal

FolderType = Literal["bush_gore", "bush_kerry"]
FileType = Literal["passive", "non_passive", "raw"]


def load_corpus(folder: FolderType, file_type: FileType = "raw") -> list[str]:
    """
    Loads the corpus from the given folder.

    Parameters
    ----------
    folder : FolderType
        The folder to load the corpus from.
    file_type : FileType, optional
        The type to load, can be passive, non-passive, or raw by default "raw"

    Returns
    -------
    list[str]
        A list of strings representing the corpus of either the raw sentences, or the splited (non)passive ones.
    """
    lines: list[str] = []
    with open(Path(__file__).parent.joinpath(folder, file_type + ".txt")) as file:
        for line in file:
            lines.append(line.strip())
    return lines


def save_sentences(
    folder: FolderType,
    passive_sentences: list[str],
    non_passive_sentences: list[str],
) -> None:
    """
    Save a list of extracted passive and non passive sentences to the given folder.

    Parameters
    ----------
    folder : FolderType
        The folder to save the sentences to.
    passive_sentences : list[str]
        The list of passive sentences to save.
    non_passive_sentences : list[str]
        The list of non passive sentences to save.
    """
    with open(Path(__file__).parent.joinpath(folder, "passive.txt"), "w") as file:
        for line in passive_sentences:
            if line:
                _ = file.write(line + "\n")

    with open(Path(__file__).parent.joinpath(folder, "non_passive.txt"), "w") as file:
        for line in non_passive_sentences:
            if line:
                _ = file.write(line + "\n")
