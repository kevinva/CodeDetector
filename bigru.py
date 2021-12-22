import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataloader

class BiGRU(nn.Module):

    def __init__(self, inputDim, embeddingDim, hiddenDim, outDim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(inputDim, embeddingDim)
        self.rnn = nn.GRU(embeddingDim, hiddenDim, bidirectional=True)
        self.fc = nn.Linear(hiddenDim * 2, outDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x = self.embedding(src)
        print('embed: ', x)
        x = x.transpose(0, 1)
        print('embed transpose: ', x)
        out, hidden = self.rnn(x)
        print('out: ', out)
        print('hidden: ', hidden)
        # print(rnnOutput.shape)
        # print(hidden)


if __name__ == '__main__':
    word2Index, index2Word, vocabSize = dataloader.getVobcabulary()
    XValid, yValid = dataloader.getData(word2Index, mode='valid')
    XValid = XValid[:2, :3]
    yValid = yValid[:2]
    print(XValid)
    print(yValid)

    src = torch.LongTensor(XValid)
    print('source: ', src)
    # x = x.transpose(0, 1)
    # print(x)
    model = BiGRU(inputDim=1000, embeddingDim=5, hiddenDim=8, outDim=8, dropout=0.5)
    model.forward(src)