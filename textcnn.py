import numpy as np
import torch
from torch import nn
from config import *


class TextCNN(nn.Module):
    
    def __init__(self, vocabSize, embeddingDim, outputSize, filterNum=OUT_CHANNEL_NUM, kernelList=KERNEL_LIST, dropout=DROPOUT):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filterNum, (kernel, embeddingDim)),
                nn.LeakyReLU(),
                nn.MaxPool2d((MAX_TOKEN_LIST_SIZE - kernel + 1, 1))
            ) for kernel in kernelList
        ])
        self.fc = nn.Linear(filterNum * len(kernelList), outputSize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
