import os
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any, Literal, Optional, cast

import torch
from attr import asdict
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.rich_model_summary import RichModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.types import STEP_OUTPUT
from rich import print
from torch import Tensor, tensor
from torch.nn.functional import l1_loss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing_extensions import Literal

from hiring_branch.trainer.config import Config, TrainConfig
from hiring_branch.trainer.loader import (
    PassiveCollator,
    PassiveDatasetModule,
    PassiveItem,
)
from hiring_branch.trainer.utils import download_weights

os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "all"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


Phases = Literal["train", "valid", "test"]


class PassiveClassifier(LightningModule):
    """
    A PytorchLightning module for classifying passive vs non passive sentences
    """

    def __del__(self) -> None:
        rmtree(self.temp_dir)

    def __init__(self, config: Config) -> None:
        self.hparams: AttributeDict
        self.wandb_logger: WandbLogger
        self.config = config
        self.temp_dir: str = mkdtemp()
        super().__init__()

        self.save_hyperparameters(asdict(config))
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        self.bert: PreTrainedModel = AutoModel.from_pretrained(config.backbone)
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

        self.clf = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                self.bert.config.hidden_size,
                self.bert.config.hidden_size // 2,
            ),
            torch.nn.Dropout(p=0.2),
            torch.nn.GELU(),
            torch.nn.Linear(
                self.bert.config.hidden_size // 2,
                1,
            ),
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.clf_loss = torch.nn.BCEWithLogitsLoss()
        # self.mse_loss = torch.nn.MSELoss()

        self.wandb_tables: dict[
            Phases, list[tuple[list[str], list[float], list[float], list[float]]]
        ] = {
            "train": [],
            "valid": [],
        }

    def _log_metrics(
        self,
        *,
        predictions: Tensor,
        references: Tensor,
        phase: Phases,
    ) -> None:
        int_predictions = torch.where(
            predictions > 0.5,
            tensor([1.0], device=references.device),
            tensor([0.0], device=references.device),
        )
        score = l1_loss(int_predictions, references)
        score_logit = l1_loss(predictions, references)
        score_one_hot = (torch.sum(references) - torch.sum(int_predictions)) / len(
            predictions
        )

        self.log_dict({f"{phase}/score": score})
        self.log_dict({f"{phase}/score_one_hot": score_one_hot})
        self.log_dict({f"{phase}/score_logits": score_logit})

    def configure_optimizers(self) -> Any:
        """
        Configure the model optimizer and the learning rate scheduler

        Returns
        -------
        A dictionary usable by pytorch lightning, c.f : https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
        """
        loaded_lr: Optional[float] = self.hparams.learning_rate
        loaded_lr_decay: Optional[float] = self.hparams.learning_rate_decay

        if loaded_lr and loaded_lr_decay:
            optimizer = AdamW(self.clf.parameters(), lr=loaded_lr)
            # optimizer = DeepSpeedCPUAdam(self.parameters(), lr=loaded_lr)
            lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=loaded_lr_decay)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "name": "linear_scheduler",
                    # "monitor": "val/loss",
                    # "strict": True,
                },
            }

        else:
            raise AssertionError(
                "Learning rate and learning rate decay needs to be defined"
            )

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        with torch.inference_mode():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        embedding = outputs.last_hidden_state.sum(dim=1).squeeze()
        clf_output = self.clf(embedding).squeeze()

        loss = self.clf_loss(clf_output, labels)
        sig_clf_output = self.sigmoid(clf_output)

        # loss = self.mse_loss(sig_clf_output, labels)
        return (sig_clf_output, loss)

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        input_ids, attention_mask, labels = batch
        logits, loss = self.forward(input_ids, attention_mask, labels)

        self.log("train/loss", loss)
        self._log_metrics(
            predictions=logits,
            references=labels,
            phase="train",
        )

        loader = cast(
            DataLoader[PassiveItem],
            self.trainer._data_connector._train_dataloader_source.dataloader(),  # type:ignore
        )
        if batch_idx % (len(loader) // 3) == 0:
            int_predictions = torch.where(
                logits > 0.5,
                tensor([1.0], device=logits.device),
                tensor([0.0], device=logits.device),
            )
            self.wandb_tables["train"].append(
                (
                    self.tokenizer.batch_decode(input_ids),
                    cast(list[float], logits.tolist()),
                    cast(list[float], int_predictions.tolist()),
                    cast(list[float], labels.tolist()),
                )
            )

        return loss

    # def on_train_epoch_end(self) -> None:
    #     self.wandb_logger.log_table(
    #         key="train/examples",
    #         columns=["Input", "Logits", "Output", "Expected"],
    #         data=self.wandb_tables["train"],
    #     )

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        """
        Same as the training step except that the model is freezed and
        we run a beam search (instead of a greedy one) to get better accuracy.
        """
        input_ids, attention_mask, labels = batch
        with torch.inference_mode():
            logits, loss = self.forward(input_ids, attention_mask, labels)

        self.log("val/loss", loss)
        int_predictions = torch.where(
            logits > 0.5,
            tensor([1.0], device=logits.device),
            tensor([0.0], device=logits.device),
        )
        self._log_metrics(
            predictions=logits,
            references=labels,
            phase="valid",
        )

        loader = cast(
            DataLoader[PassiveItem],
            self.trainer._data_connector._val_dataloader_source.dataloader(),  # type:ignore
        )
        if batch_idx % (len(loader) // 3) == 0:
            self.wandb_tables["valid"].append(
                (
                    self.tokenizer.batch_decode(input_ids),
                    cast(list[float], logits.tolist()),
                    cast(list[float], int_predictions.tolist()),
                    cast(list[float], labels.tolist()),
                )
            )

    # def on_validation_epoch_end(self) -> None:
    #     self.wandb_logger.log_table(
    #         key="valid/examples",
    #         columns=["Input", "Logits", "Output", "Expected"],
    #         data=self.wandb_tables["valid"],
    #     )

    def sanity_check(self, train_config: TrainConfig) -> None:
        data_module = PassiveDatasetModule(
            train_folder=train_config.train_folder,
            valid_folder=train_config.valid_folder,
            batch_size=train_config.train_batch_size,
            valid_batch_size=train_config.valid_batch_size,
            collator=PassiveCollator(tokenizer=self.tokenizer),
        )
        self.hparams.learning_rate = train_config.learning_rate
        self.hparams.learning_rate_decay = train_config.learning_rate_decay
        self.wandb_logger = WandbLogger(
            offline=True,
            project="spellchecker",
            entity=self.config.entity,
        )

        trainer = Trainer(fast_dev_run=True)
        trainer.fit(model=self, datamodule=data_module)

    def launch_train(self, train_config: TrainConfig) -> None:
        if train_config.checkpoint:
            train_config.checkpoint = str(download_weights(train_config.checkpoint))

        self.wandb_logger = WandbLogger(
            project="hiring_branch",
            log_model="all",
            entity=self.config.entity,
            # id=run_id,
        )
        self.wandb_logger.watch(self.clf, log_freq=50)

        trainer = Trainer(
            logger=self.wandb_logger,
            enable_checkpointing=True,
            default_root_dir="./runs",
            accelerator="gpu",
            # devices=2,
            # strategy="deepspeed_stage_2_offload",
            callbacks=[
                RichProgressBar(),
                RichModelSummary(max_depth=3),
                ModelCheckpoint(
                    monitor="valid/score",
                    mode="min",
                    every_n_epochs=1,
                    # save_on_train_epoch_end=True,
                ),
                LearningRateMonitor(),
            ],
            gpus=[self.config.cuda_device],
            # gpus=[0],
            # accumulate_grad_batches=train_config.gradient_accumulation_steps,
            max_epochs=train_config.epochs,
            log_every_n_steps=train_config.logging_steps or 50,
            # precision="bf16",
            precision=16 if train_config.precision == 16 else train_config.precision,
            check_val_every_n_epoch=train_config.validation_check_every,
        )

        data_module = PassiveDatasetModule(
            train_folder=train_config.train_folder,
            valid_folder=train_config.valid_folder,
            batch_size=train_config.train_batch_size,
            valid_batch_size=train_config.valid_batch_size,
            collator=PassiveCollator(tokenizer=self.tokenizer),
        )

        self.hparams.learning_rate = train_config.learning_rate
        self.hparams.learning_rate_decay = train_config.learning_rate_decay
        self.wandb_logger.log_hyperparams(asdict(train_config))

        print(f"[cyan]Parameters for training:[/cyan] \n{self.hparams}")
        trainer.fit(
            model=self,
            datamodule=data_module,
            ckpt_path=train_config.checkpoint,
        )

    @staticmethod
    def optimize(checkpoint: str) -> None:
        if not Path(checkpoint).exists():
            checkpoint_file = download_weights(checkpoint)
        else:
            checkpoint_file = checkpoint
        model = PassiveClassifier.load_from_checkpoint(str(checkpoint_file))

        checkpoint_folder = Path(__file__).parent.parent.joinpath("models")
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        model_file = checkpoint_folder.joinpath("classifier.onnx")
        model.clf.eval()

        max_bert_length: int = model.bert.config.max_length
        bert_embed_size: int = model.bert.config.hidden_size

        with open(model_file, "wb") as f:
            torch.onnx.export(
                model.clf,
                torch.ones((max_bert_length, bert_embed_size)),
                f,
                export_params=True,
                input_names=["embedding"],
                output_names=["logits"],
                dynamic_axes={"embedding": {0: "batch_size"}},
            )

        model_checkpoint = model.config.backbone
        with TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            onnx_save_directory = tmp_dir_path.joinpath("onnx")
            file_name = "model.onnx"
            quantized_file_name = "model_quantized.onnx"
            onnx_quantized_model_path = onnx_save_directory.joinpath(
                quantized_file_name
            )
            onnx_model_path = onnx_save_directory.joinpath(file_name)
            onnx_chad_model_path = onnx_save_directory.joinpath("model_chad.onnx")
            feature = "default"

            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = ORTModelForFeatureExtraction.from_pretrained(
                model_checkpoint, from_transformers=True
            )

            model.save_pretrained(onnx_save_directory, file_name=file_name)
            tokenizer.save_pretrained(onnx_save_directory)
            del model

            quantizer = ORTQuantizer.from_pretrained(model_checkpoint, feature=feature)
            quantizer.export(
                onnx_model_path=onnx_model_path,
                onnx_quantized_model_output_path=onnx_quantized_model_path,
                quantization_config=AutoQuantizationConfig.avx2(
                    is_static=False, per_channel=True
                ),
            )
            del quantizer

            optimizer = ORTOptimizer.from_pretrained(model_checkpoint, feature=feature)
            optimizer.export(
                onnx_model_path=onnx_quantized_model_path,
                onnx_optimized_model_output_path=onnx_chad_model_path,
                optimization_config=OptimizationConfig(
                    optimization_level=99, optimize_for_gpu=False
                ),
            )
            del optimizer

            model = ORTModelForFeatureExtraction.from_pretrained(
                onnx_chad_model_path.parent
            )
            model.save_pretrained(model_file.parent.joinpath("bert"))
            tokenizer.save_pretrained(model_file.parent.joinpath("bert"))
