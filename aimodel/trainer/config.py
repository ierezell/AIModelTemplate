from pathlib import Path
from typing import Literal, Self, cast

from attr import Attribute
from attrs import define, field
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
    run_id: str | None = field(default=None)

    # The seed to replicate the run
    seed: int = field(default=42)

    # The Gpu id on which to run. -1 is Cpu, >= 0 is the cuda index
    cuda_device: int = field(default=0 if is_available() else -1)

    @classmethod
    def from_dict(cls: type[Self], any_dict: dict[str, object]) -> "Config":
        return cls(
            backbone=cast(str, any_dict.get("backbone")),
            entity=cast(str, any_dict.get("entity")),
            project=cast(str, any_dict.get("project")),
            run_id=cast(str, any_dict.get("run_id")),
            seed=cast(int, any_dict.get("seed", 42)),
            cuda_device=cast(int, any_dict.get("cuda_device", -1)),
        )


def _path_is_valid(
    instance: object,  # noqa: ARG001 (unused)
    attribute: "Attribute[Path]",  # noqa: ARG001 (unused)
    value: Path,
) -> None:
    msg: str | None = None
    if not Path(value).exists():
        msg = f"{value} does not exist"

    if not Path(value).is_dir():
        msg = f"{value} is not a directory"

    if msg:
        raise ValueError(msg)

    files = [p.name for p in Path(value).iterdir()]
    if "passive.txt" not in files:
        msg = f"{value} does not contains passive.txt"

    if "non_passive.txt" not in files:
        msg = f"{value} does not contains non_passive.txt"

    if msg:
        raise ValueError(msg)


@define
class TrainConfig:
    train_folder: Path = field(default=Path("data/bush_gore"))
    valid_folder: Path = field(
        default=Path(__file__).parent.parent.joinpath("data/bush_kerry"),
        validator=_path_is_valid,
    )

    # The learning rate starting value
    learning_rate: float = field(default=1e-3)

    # The learning rate starting value
    learning_rate_decay: float = field(default=0.95)

    # Number of epochs to train over
    epochs: int = field(default=15)

    # The number of warmup step. Check https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_schedule_with_warmup
    warmup_steps: int = field(default=100)

    # How much steps to wait before propagating the gradient.
    # This simulate a bigger batch size
    gradient_accumulation_steps: int = field(default=4)

    # The number of steps to log
    logging_steps: int | None = field(default=None)

    # Size of each mini-batch when validating
    valid_batch_size: int = field(default=2)

    # Size of each mini-batch when training
    train_batch_size: int = field(default=2)
    precision: Literal[16, 32, 64] = field(default=32)

    checkpoint: str | None = field(default=None)

    validation_check_every: int = field(default=1)

    @classmethod
    def from_dict(cls: type[Self], any_dict: dict[str, object]) -> "TrainConfig":
        return cls(
            train_folder=Path(
                cast(str, any_dict.get("train_folder", "data/bush_gore")),
            ),
            valid_folder=Path(
                cast(str, any_dict.get("valid_folder", "data/bush_kerry")),
            ),
            learning_rate=cast(float, any_dict.get("learning_rate")),
            learning_rate_decay=cast(float, any_dict.get("learning_rate_decay")),
            epochs=cast(int, any_dict.get("epochs")),
            warmup_steps=cast(int, any_dict.get("warmup_steps")),
            gradient_accumulation_steps=cast(
                int,
                any_dict.get("gradient_accumulation_steps", 4),
            ),
            logging_steps=cast(int, any_dict.get("logging_steps")),
            valid_batch_size=cast(int, any_dict.get("valid_batch_size")),
            train_batch_size=cast(int, any_dict.get("train_batch_size")),
            precision=cast(Literal[16, 32, 64], any_dict.get("precision", 32)),
            checkpoint=cast(str | None, any_dict.get("checkpoint")),
            validation_check_every=cast(int, any_dict.get("validation_check_every", 1)),
        )
