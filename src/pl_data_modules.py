from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from src.wsd.dataset import SimpleTextDataset
import conf
import os

class DataModule(pl.LightningDataModule):
    def __init__(self, train_path: str, val_path: str, test_path: str, batch_size: int):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.tokenizer_name = conf.pmodel["name"]
        self.max_len = conf.pmodel["max_len"]
        self.example_sentences = conf.example_sentences
        self.dataset_name = None  # 数据集名称

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SimpleTextDataset(
                self.train_path,
                tokenizer_name=self.tokenizer_name,
                max_len=self.max_len,
                example_sentences=self.example_sentences,
                is_train=True,
            )
            self.val_dataset = SimpleTextDataset(
                self.val_path,
                tokenizer_name=self.tokenizer_name,
                max_len=self.max_len,
                example_sentences=self.example_sentences,
            )
        if stage == "test" or stage is None:
            self.dataset_name = os.path.basename(self.test_path).split('_')[0]  # 提取数据集名称
            print(f"Dataset name set to: {self.dataset_name}")
            self.test_dataset = SimpleTextDataset(
                self.test_path,
                tokenizer_name=self.tokenizer_name,
                max_len=self.max_len,
                example_sentences=self.example_sentences,
                dataset_name=self.dataset_name  # 传递数据集名称
            )


    def collate_fn(self, batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[2] for item in batch])
        return input_ids, attention_mask, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        dataloader.dataset_name = self.dataset_name  # 将 dataset_name 存储在 DataLoader 中
        return dataloader

    def set_dataset_name(self, model):
        """将 dataset_name 传递给模型"""
        model.dataset_name = self.dataset_name