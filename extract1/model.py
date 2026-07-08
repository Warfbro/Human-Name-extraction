"""ERINE + CRF 模型"""
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer

from config import ERNIE_LOCAL, BIO_LABELS, CHECKPOINT, CHECKPOINT_FROZEN, CHECKPOINT_FC2


class ErnieCRF(nn.Module):
    """ERNIE 3.0 encoder + Linear + CRF"""

    def __init__(self, model_path, num_labels):
        super().__init__()
        self.ernie = AutoModel.from_pretrained(model_path)
        self.fc = nn.Linear(self.ernie.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        mask = attention_mask.bool()
        hidden = self.ernie(input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.fc(hidden)
        if labels is not None:
            return -self.crf(emissions, labels, mask=mask, reduction="mean")
        return self.crf.decode(emissions, mask=mask)


class ErnieCRF2(nn.Module):
    """ERNIE 3.0 encoder + 双层FC(hidden→3*hidden→3) + CRF"""

    def __init__(self, model_path, num_labels, hidden_factor=3):
        super().__init__()
        self.ernie = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.ernie.config.hidden_size
        mid_size = self.hidden_size * hidden_factor
        self.fc1 = nn.Linear(self.hidden_size, mid_size)
        self.fc2 = nn.Linear(mid_size, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        mask = attention_mask.bool()
        hidden = self.ernie(input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(self.relu(self.fc1(hidden)))
        emissions = self.fc2(x)
        if labels is not None:
            return -self.crf(emissions, labels, mask=mask, reduction="mean")
        return self.crf.decode(emissions, mask=mask)


def load_model(device="cuda", frozen=False, fc2=False):
    """加载训练好的模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(ERNIE_LOCAL)
    if fc2:
        model = ErnieCRF2(ERNIE_LOCAL, len(BIO_LABELS), hidden_factor=2).to(device)
        ckpt = CHECKPOINT_FC2
    else:
        model = ErnieCRF(ERNIE_LOCAL, len(BIO_LABELS)).to(device)
        ckpt = CHECKPOINT_FROZEN if frozen else CHECKPOINT
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model, tokenizer
