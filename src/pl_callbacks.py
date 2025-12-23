from pytorch_lightning.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score


class ClassificationMetricsCallback(Callback):
    """
    A callback to calculate and log classification metrics (accuracy, F1 score) during validation.
    """

    def __init__(self):
        self.predictions = []
        self.targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Collect predictions and targets after each validation batch.
        """
        if outputs is None:
            return
        logits = outputs["logits"]
        labels = outputs["labels"]
        preds = logits.argmax(dim=1).cpu().tolist()
        self.predictions.extend(preds)
        self.targets.extend(labels.cpu().tolist())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Calculate metrics and log them after each validation epoch.
        """
        accuracy = accuracy_score(self.targets, self.predictions)
        f1 = f1_score(self.targets, self.predictions, average="weighted")

        # Log metrics
        pl_module.log("val_accuracy", accuracy, prog_bar=True)
        pl_module.log("val_f1", f1, prog_bar=True)

        # Clear stored predictions and targets
        self.predictions.clear()
        self.targets.clear()
