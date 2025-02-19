import os
from pathlib import Path
from typing import Self, TypedDict, cast, override

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.cuda import is_available as is_gpu_available
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class PassiveItem(TypedDict):
    sentence: str
    label: int


class PassiveDataset(Dataset[PassiveItem]):
    def __init__(self: Self, data_folder: Path) -> None:
        super().__init__()
        self.data: list[PassiveItem] = []

        with open(data_folder.joinpath("passive.txt")) as f:
            for line in f:
                self.data.append(PassiveItem(sentence=line.strip(), label=1))

        number_non_passive = len(self.data)
        with open(data_folder.joinpath("non_passive.txt")) as f:
            for line in f:
                if number_non_passive > 0:
                    self.data.append(PassiveItem(sentence=line.strip(), label=0))
                    number_non_passive -= 1

    def __len__(self: Self) -> int:
        return len(self.data)

    @override
    def __getitem__(self: Self, idx: int) -> PassiveItem:
        return self.data[idx]


class PassiveCollator:
    def __init__(self: Self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self: Self, x: list[PassiveItem]) -> tuple[Tensor, Tensor, Tensor]:
        if not self.tokenizer.cls_token:
            self.tokenizer.cls_token = self.tokenizer.pad_token

        model_input = self.tokenizer(
            [i["sentence"] for i in x],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        model_input["labels"] = torch.tensor(
            [i["label"] for i in x],
            dtype=torch.float,
        )

        return (
            cast(Tensor, model_input["input_ids"]),
            cast(Tensor, model_input["attention_mask"]),
            cast(Tensor, model_input["labels"]),
        )


class PassiveDatasetModule(LightningDataModule):
    def __init__(
        self: Self,
        *,
        train_folder: Path,
        valid_folder: Path,
        batch_size: int,
        valid_batch_size: int,
        collator: PassiveCollator,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.seed = seed
        self.train_dataset = PassiveDataset(data_folder=train_folder)
        self.valid_dataset = PassiveDataset(data_folder=valid_folder)
        self.collator = collator

    @override
    def train_dataloader(self: Self) -> DataLoader[PassiveItem]:
        train_loader: DataLoader[PassiveItem] = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 8,
            pin_memory=is_gpu_available(),
            collate_fn=self.collator,
            drop_last=True,
            worker_init_fn=None,
        )
        return train_loader

    @override
    def val_dataloader(self: Self) -> DataLoader[PassiveItem]:
        valid_loader: DataLoader[PassiveItem] = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 8,
            pin_memory=is_gpu_available(),
            drop_last=True,
            collate_fn=self.collator,
            worker_init_fn=None,
        )
        return valid_loader
