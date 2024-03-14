from pathlib import Path
from typing import Literal, Self

from attr import Attribute, attrib, define
from torch.cuda import is_available


@define
class Config:
    # The huggingface model name
    backbone: str

    # The weight and biases entity to use
    entity: str = "MY_NAME"

    # The weight and biases project to use
    project: str = "MY_PROJECT"

    # The Weight and Biases run to load or save to
    run_id: str | None = attrib(default=None)

    # The seed to replicate the run
    seed: int = attrib(default=42)

    # The Gpu id on which to run. -1 is Cpu, >= 0 is the cuda index
    cuda_device: int = attrib(default=0 if is_available() else -1)

    @classmethod
    def from_dict(cls: type[Self], any_dict: dict[str, object]) -> "Config":
        return cls(
            backbone=any_dict.get("backbone"),
            entity=any_dict.get("entity"),
            project=any_dict.get("project"),
            run_id=any_dict.get("run_id"),
            seed=any_dict.get("seed", 42),
            cuda_device=any_dict.get("cuda_device", -1),
        )


def _path_is_valid(instance: object, attribute: "Attribute[Path]", value: Path) -> None:
    if not Path(value).exists():
        raise ValueError(f"{value} does not exist")

    if not Path(value).is_dir():
        raise ValueError(f"{value} is not a directory")

    files = list(p.name for p in Path(value).iterdir())
    if "passive.txt" not in files:
        raise ValueError(f"{value} does not contains passive.txt")

    if "non_passive.txt" not in files:
        raise ValueError(f"{value} does not contains non_passive.txt")


@define
class TrainConfig:
    train_folder: Path = attrib(default=Path("data/bush_gore"))
    valid_folder: Path = attrib(
        default=Path(__file__).parent.parent.joinpath("data/bush_kerry"),
        validator=_path_is_valid,
    )

    # The learning rate starting value
    learning_rate: float = attrib(default=1e-3)

    # The learning rate starting value
    learning_rate_decay: float = attrib(default=0.95)

    # Number of epochs to train over
    epochs: int = attrib(default=15)

    # The number of warmup step. Check https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_schedule_with_warmup
    warmup_steps: int = attrib(default=100)

    # How much steps to wait before propagating the gradient. This simulate a bigger batch size
    gradient_accumulation_steps: int = attrib(default=4)

    # The number of steps to log
    logging_steps: int | None = attrib(
        default=None,
    )

    # Size of each mini-batch when validating
    valid_batch_size: int = attrib(
        default=2,
    )

    # Size of each mini-batch when training
    train_batch_size: int = attrib(
        default=2,
    )
    precision: Literal[16, 32, 64] = attrib(default=32)

    checkpoint: str | None = attrib(default=None)

    validation_check_every: int = attrib(
        default=1,
    )

    @classmethod
    def from_dict(cls, any_dict: dict[str, Any]) -> "TrainConfig":
        return cls(**any_dict)
