from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from aimodel.trainer.loader import PassiveCollator, PassiveDatasetModule


def test_loader():
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    dm = PassiveDatasetModule(
        train_folder=Path(__file__).parent.parent.joinpath("data/bush_gore"),
        valid_folder=Path(__file__).parent.parent.joinpath("data/bush_kerry"),
        batch_size=2,
        valid_batch_size=2,
        collator=PassiveCollator(tokenizer),
        seed=123,
    )
    train_loader = dm.train_dataloader()

    for inputs, _, labels in train_loader:
        assert inputs.size()[0] == 2
        assert labels.size() == (2,)

    valid_loader = dm.val_dataloader()

    for inputs, _, labels in valid_loader:
        assert inputs.size()[0] == 2
        assert labels.size() == (2,)
