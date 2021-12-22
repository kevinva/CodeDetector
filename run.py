import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from config import *
import textcnn
import dataloader
import json
import time

def binaryAcc(preds, y):
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

def getConfustionMaxtrixData(preds, y):
    assert preds.shape[0] == y.shape[0], 'error shape: preds = {}, y = {}'.format(preds.shape, y.shape)

    tp, tn, fp, fn = 0, 0, 0, 0
    for predItem, yItem in zip(preds, y):
        if predItem == 0 and yItem == 0:
            tn += 1
        elif predItem == 0 and yItem == 1:
            fn += 1
        elif predItem == 1 and yItem == 1:
            tp += 1
        elif predItem == 1 and yItem == 0:
            fp += 1
    return tp, tn, fp, fn

def train(model, XTrain, yTrain, optimizer, lossFunction, useGPU):
    avgAcc = []
    model.train()

    for xBatch, yBatch in dataloader.getBatch(XTrain, yTrain):
        xBatch = torch.LongTensor(xBatch)
        yBatch = torch.tensor(yBatch).long()
        yBatch = yBatch.squeeze()
        if useGPU:
            xBatch = xBatch.cuda()
            yBatch = yBatch.cuda()

        pred = model(xBatch)

        if useGPU:
            predCPU = pred.cpu()
            yBatchCPU = yBatch.cpu()
            acc = binaryAcc(torch.max(predCPU, dim=1)[1], yBatchCPU)
            avgAcc.append(acc)
        else:
            acc = binaryAcc(torch.max(pred, dim=1)[1], yBatch)
            avgAcc.append(acc)

        loss = lossFunction(pred, yBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avgAcc = np.array(avgAcc).mean()
    return avgAcc


def evaluate(model, XTest, yTest, useGPU):
    avgAcc = []
    model.eval()
    tpCount, tnCount, fpCount, fnCount = 0, 0, 0, 0
    with torch.no_grad():
        for xBatch, yBatch in dataloader.getBatch(XTest, yTest):
            xBatch = torch.LongTensor(xBatch)
            yBatch = torch.tensor(yBatch).long().squeeze()
            if useGPU:
                xBatch = xBatch.cuda()
                yBatch = yBatch.cuda()

            pred = model(xBatch)

            if useGPU:
                predCPU = pred.cpu()
                yBatchCPU = yBatch.cpu()
                acc = binaryAcc(torch.max(predCPU, dim=1)[1], yBatchCPU)
                avgAcc.append(acc)
                tp, tn, fp, fn = getConfustionMaxtrixData(torch.max(predCPU, dim=1)[1], yBatchCPU)
            else:
                acc = binaryAcc(torch.max(pred, dim=1)[1], yBatch)
                avgAcc.append(acc)
                tp, tn, fp, fn = getConfustionMaxtrixData(torch.max(pred, dim=1)[1], yBatch)

            tpCount += tp
            tnCount += tn
            fpCount += fp
            fnCount += fn
            avgAcc.append(acc)

    avgAcc = np.array(avgAcc).mean()
    return avgAcc, tpCount, tnCount, fpCount, fnCount


if __name__ == '__main__':
    word2Index, index2Word, vocabSize = dataloader.getVobcabulary()
    XTrain, yTrain = dataloader.getData(word2Index, mode='train')
    XTest, yTest = dataloader.getData(word2Index, mode='test')
    XValid, yValid = dataloader.getData(word2Index, mode='valid')

    model = textcnn.TextCNN(vocabSize, EMBEDDING_SIZE, 2)
    criterion = nn.CrossEntropyLoss()

    canUseGPU = torch.cuda.is_available()
    print('Can use GPU: {}'.format(canUseGPU))
    if canUseGPU:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print('Model structure: ', model)

    modelTrainAcc, modelTestAcc = [], []
    tpCount, tnCount, fpCount, fnCount = 0, 0, 0, 0
    for epoch in range(EPOCH):
        trainAcc = train(model, XTrain, yTrain, optimizer, criterion, canUseGPU)
        print('epoch = {}, train accuracy = {}'.format(epoch, trainAcc))

        testAcc, tpCount, tnCount, fpCount, fnCount = evaluate(model, XTest, yTest, canUseGPU)
        print('epoch = {}, test accuracy = {}'.format(epoch, testAcc))

        modelTrainAcc.append(trainAcc.item())
        modelTestAcc.append(testAcc.item())
    
    trainAccFilePath = './output/train_accuracy.json'
    testAccFilePath = './output/test_accuracy.json'
    testConfusionMatrixFilePath = './output/test_confusion_matrix.json'
    modelStateFilePath = './output/hoho_code_detector_model_{}.pth'.format(int(time.time()))

    torch.save({'model': model.state_dict()}, modelStateFilePath)

    with open(trainAccFilePath, 'w') as file1:
        modelTrainAccStr = json.dumps(modelTrainAcc)
        file1.write(modelTrainAccStr)

    with open(testAccFilePath, 'w') as file2:
        modelTestAccStr = json.dumps(modelTestAcc)
        file2.write(modelTestAccStr)

    with open(testConfusionMatrixFilePath, 'w') as file3:
        cmDict = dict({'TP': tpCount, 'TN': tnCount, 'FP': fpCount, 'FN': fnCount})
        cmDictStr = json.dumps(cmDict)
        file3.write(cmDictStr)

    plt.plot(modelTrainAcc)
    plt.plot(modelTestAcc)

    plt.ylim(ymin=0.5, ymax=1.01)
    plt.title("The accuracy of textCNN model")
    plt.legend(["train", 'test'])
    plt.show()
