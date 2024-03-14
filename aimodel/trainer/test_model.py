from pathlib import Path

from aimodel.trainer.bert_model import PassiveClassifier
from aimodel.trainer.config import Config, TrainConfig


def test_model():
    p = PassiveClassifier(config=Config(backbone="bert-base-uncased"))
    p.sanity_check(
        TrainConfig(
            train_folder=Path(__file__).parent.parent.joinpath("data/bush_gore")
        )
    )
