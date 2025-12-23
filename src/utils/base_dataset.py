import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class BaseDataset:
    def __init__(self, file_path: str, tokenizer_name: str, max_len: int):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        # Load data from file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence, label = line.strip().split('\t')
                self.data.append((sentence, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        tokenized = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        return tokenized["input_ids"].squeeze(0), tokenized["attention_mask"].squeeze(0), label

    def collate_fn(self, batch):
        """
        Custom collate function to process batches.
        """
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[2] for item in batch])
        return input_ids, attention_mask, labels
