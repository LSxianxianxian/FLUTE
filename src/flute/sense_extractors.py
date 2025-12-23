import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MetaphorClassifier(nn.Module):
    """
    A simple Transformer-based classifier for metaphor detection.
    """
    def __init__(self, model_name: str, num_classes: int = 4, dropout: float = 0.1):
        super(MetaphorClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)


    #def forward(self, input_ids, attention_mask):
       # outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
       # pooled_output = outputs.last_hidden_state[:, 0]  # CLS token representation
       # pooled_output = self.dropout(pooled_output)
       # logits = self.classifier(pooled_output)
       # return logits
    
    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # CLS token representation
        cls_output = self.dropout(cls_output)

        if return_embedding:
            return cls_output  # 用于对比学习
        else:
            logits = self.classifier(cls_output)
            return logits
