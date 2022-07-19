from pathlib import Path

from hiring_branch.trainer.bert_model import PassiveClassifier
from hiring_branch.trainer.config import Config, TrainConfig


def test_model():
    p = PassiveClassifier(config=Config(backbone="bert-base-uncased"))
    p.sanity_check(
        TrainConfig(
            train_folder=Path(__file__).parent.parent.joinpath("data/bush_gore")
        )
    )
