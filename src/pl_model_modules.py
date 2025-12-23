import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from src.wsd.sense_extractors import MetaphorClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import conf

class MetaphorModel(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int = 4, lr: float = 3e-5, contrastive_weight: float = 0.25):
        super(MetaphorModel, self).__init__()
        self.model = MetaphorClassifier(model_name=model_name, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.contrastive_weight = contrastive_weight
        self.temperature = 0.28  # 可调参数
        self.test_results = defaultdict(lambda: {"preds": [], "labels": []})

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def supervised_contrastive_loss(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=self.device)
        mask = mask * logits_mask

        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        logits = self(input_ids, attention_mask)
        ce_loss = self.criterion(logits, labels)

        embeddings = self.model(input_ids, attention_mask, return_embedding=True)
        contrastive_loss = self.supervised_contrastive_loss(embeddings, labels)

        loss = ce_loss + self.contrastive_weight * contrastive_loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return {"logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)

        dataset_name = getattr(self, "dataset_name", "unknown")
        self.test_results[dataset_name]["preds"].append(preds.cpu())
        self.test_results[dataset_name]["labels"].append(labels.cpu())

        return (preds == labels).float().mean()

    def test_epoch_end(self, outputs):
        for dataset_name, results in self.test_results.items():
            all_preds = torch.cat(results["preds"], dim=0).numpy()
            all_labels = torch.cat(results["labels"], dim=0).numpy()
            accuracy = (all_preds == all_labels).mean()

            label_set = sorted(set(all_labels.tolist()))
            if dataset_name == "mixed" or len(label_set) > 2:
                precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
                recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
                f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            else:
                main_label = max(label_set)
                cleaned_preds = [p if p == main_label else 0 for p in all_preds]
                precision = precision_score(all_labels, cleaned_preds, pos_label=main_label, average="binary", zero_division=0)
                recall = recall_score(all_labels, cleaned_preds, pos_label=main_label, average="binary", zero_division=0)
                f1 = f1_score(all_labels, cleaned_preds, pos_label=main_label, average="binary", zero_division=0)

            self.log(f"test_accuracy/{dataset_name}", accuracy, prog_bar=True)
            self.log(f"test_precision/{dataset_name}", precision, prog_bar=True)
            self.log(f"test_recall/{dataset_name}", recall, prog_bar=True)
            self.log(f"test_f1/{dataset_name}", f1, prog_bar=True)

            print(f"\n🔹 Dataset {dataset_name} - Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        self.test_results.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=conf.weight_decay)
        return optimizer
