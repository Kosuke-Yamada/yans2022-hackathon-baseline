# -*-coding:utf-8-*-

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer):

        self.out_inputs = []
        for df_dict in df.to_dict("records"):
            inputs = tokenizer(df_dict["review_body"], truncation=True)
            if "label" in df_dict:
                inputs.update({"label": df_dict["label"]})
            self.out_inputs.append(inputs)

    def __len__(self):
        return len(self.out_inputs)

    def __getitem__(self, idx):
        return self.out_inputs[idx]


class ReviewDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.df = pd.read_json(hparams.input_file, orient="records", lines=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        self.batch_size = hparams.batch_size

    def _make_label(self):
        self.df.loc[:, "label"] = self.df.loc[:, "helpful_votes"].apply(
            lambda x: np.log(x + 1)
        )

    def setup(self, stage):
        if stage == "fit":
            self._make_label()
            df_train = self.df[self.df["sets"] == "training-train"]
            df_val = self.df[self.df["sets"] == "training-val"]

            self.train_dataset = ReviewDataset(df_train, self.tokenizer)
            self.val_dataset = ReviewDataset(df_val, self.tokenizer)
        elif stage == "test":
            self.test_dataset = ReviewDataset(self.df, self.tokenizer)
        elif stage == "predict":
            self.predict_dataset = ReviewDataset(self.df, self.tokenizer)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, False)

    def predict_dataloader(self):
        return self._get_dataloader(self.predict_dataset, False)

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=shuffle,
        )

    def _collate_fn(self, batch):
        output_dict = {}
        for i in ["input_ids", "token_type_ids", "attention_mask"]:
            output_dict[i] = nn.utils.rnn.pad_sequence(
                [torch.LongTensor(b[i]) for b in batch],
                batch_first=True,
            )
        if "label" in batch[0]:
            output_dict["label"] = torch.FloatTensor([[b["label"]] for b in batch])
        return output_dict


class ReviewRegressionNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ReviewRegressionNet, self).__init__()
        self.save_hyperparameters(hparams)

        self.pretrained_model = AutoModel.from_pretrained(self.hparams.model_name)
        self.fc = nn.Linear(self.pretrained_model.config.hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        output = self.pretrained_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )["last_hidden_state"][:, 0, :]
        return self.fc(output)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def _compute_rmse(self, output, true):
        return torch.sqrt(self.criterion(output, true))

    def training_step(self, batch, _):
        output = self.forward(batch)
        loss = self.criterion(output, batch["label"])
        rmse = self._compute_rmse(output, batch["label"])
        return {"loss": loss, "rmse": rmse}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        self.log("train_loss", avg_loss.detach(), on_epoch=True, prog_bar=True)
        self.log("train_rmse", avg_rmse.detach(), on_epoch=True, prog_bar=True)

    def validation_step(self, batch, _):
        output = self.forward(batch)
        loss = self.criterion(output, batch["label"])
        rmse = self._compute_rmse(output, batch["label"])
        return {"loss": loss, "rmse": rmse}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        self.log("val_loss", avg_loss.detach(), on_epoch=True, prog_bar=True)
        self.log("val_rmse", avg_rmse.detach(), on_epoch=True, prog_bar=True)
        return {"val_loss": avg_loss, "val_rmse": avg_rmse}

    def test_step(self, batch, _):
        output = self.forward(batch)
        loss = self.criterion(output, batch["label"])
        rmse = self._compute_rmse(output, batch["label"])
        return {"loss": loss, "rmse": rmse}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        self.log("test_loss", avg_loss.detach(), on_epoch=True, prog_bar=True)
        self.log("test_rmse", avg_rmse.detach(), on_epoch=True, prog_bar=True)
        return {"test_loss": avg_loss, "test_rmse": avg_rmse}

    def predict_step(self, batch, _):
        return self.forward(batch)
