import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from config import *

FUNCTION_ID_PREFIX = 'hoho_func'
VAR_ID_PREFIX = 'hoho_var'
POINTER_TYPE_ID_PREFIX = 'hoho_pointer_type'
VAR_TYPE_ID_PREFIX = 'hoho_var_type'
STRING_CONSTANT_PREFIX = 'hoho_str_constant'
NUMERIC_CONSTANT_PREFIX = 'hoho_numeric_constant'
C_VAR_TYPE_LIST = ['char', 'wchar_t', 'short', 'int', 'int8_t', 'int16_t', 
                   'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 
                   'uint64_t', 'float', 'double', 'long', 'size_t', 'bool'] 
UNKNOWN_TOKEN = '<unk>'
PADDING_TOKEN = '<pad>'


def getTokenWithFile(filePath):
    tokenList = list()
    wrongFilePath = './tmp/wrong_{}.txt'.format(int(time.time()))
    isWrong = False
    with open(filePath, 'r') as file:
        lineList = file.readlines()
        isLastType = False
        for index, line in enumerate(lineList):
            items = line.split('\t')
            if len(items) > 0:
                subItems = items[0].split(' ')
                if len(subItems) > 1:
                    val = subItems[1][1:][:-1]   # [1:]为去掉前单引号，[:-1]为去掉后单引号

                    if subItems[0] == 'identifier':
                        if isLastType:
                            isLastType = False
                            tokenList.append(VAR_ID_PREFIX)
                            continue

                        if index + 1 < len(lineList):
                            nextItem = lineList[index + 1]
                            nextSubItems = nextItem.split(' ')
                            if len(nextSubItems) > 1:
                                if nextSubItems[0] == 'l_paren':
                                    # tokenList.append('{}:{}'.format(FUNCTION_ID_PREFIX, subItems[1]))  # 自定义函数名
                                    tokenList.append(FUNCTION_ID_PREFIX)  # 自定义函数名
                                elif nextSubItems[0] == 'star':
                                    # tokenList.append('{}:{}'.format(POINTER_TYPE_ID_PREFIX, subItems[1]))  # 指针类型
                                    tokenList.append(POINTER_TYPE_ID_PREFIX)  # 指针类型
                                elif nextSubItems[0] == 'identifier':
                                    if val in C_VAR_TYPE_LIST:
                                        tokenList.append(val)
                                    else:
                                        # tokenList.append('{}:{}'.format(VAR_TYPE_ID_PREFIX, subItems[1]))  # 用户自定义变量类型
                                        tokenList.append(VAR_TYPE_ID_PREFIX)  # 用户自定义变量类型
                                    isLastType = True
                                else:
                                    tokenList.append(val)   
                            else:
                                tokenList.append(val)   
                        else:
                            tokenList.append(val) 
                    elif subItems[0] == 'numeric_constant':
                        tokenList.append(NUMERIC_CONSTANT_PREFIX)
                    elif subItems[0] == 'string_literal':
                        tokenList.append(STRING_CONSTANT_PREFIX)
                    else:
                        tokenList.append(val)
   
                else:
                    with open(wrongFilePath, 'w') as wrongFile:
                        wrongFile.writelines(lineList)
                    isWrong = True
                    break
            else:
                with open(wrongFilePath, 'w') as wrongFile:
                    wrongFile.writelines(lineList)
                isWrong = True
                break

    if isWrong:
        return list()
    else:   
        return tokenList


def readData(mode='train'):
    path = ''
    if mode == 'train':
        path = './data/train.jsonl'
    elif mode == 'test':
        path = './data/test.jsonl'
    else:
        path = './data/valid.jsonl'

    sampleList = list()
    with open(path, 'r') as file:
        for line in file.readlines():
            codeDict = json.loads(line)
            if codeDict is not None:
                sampleList.append(codeDict)

    dataList = list()
    targetList = list()
    for index, codeDict in enumerate(sampleList):
        codeStr = codeDict.get('func', '')
        target = codeDict.get('target', 0)
        if len(codeStr) > 0:
            filePath = './tmp/code_{}.c'.format(index % 10)
            with open(filePath, 'w') as codeFile:
                codeFile.write(codeStr)

            tmpTokenFilePath = './tmp/tokens_{}.txt'.format(index % 10)
            os.system('clang -fsyntax-only -Xclang -dump-tokens {} >& {}'.format(filePath, tmpTokenFilePath))

            statement = getTokenWithFile(tmpTokenFilePath)
            if len(statement) > 0:
                targetList.append(target)
                dataList.append(statement)
            
            # if index == 2:
            #     break  # hoho_debug
        print('finish count: {}'.format(index))  # hoho_debug

    assert len(dataList) == len(targetList), 'error length!'
    
    print('original samples count: {}'.format(len(sampleList)))
    print('samples after cleaned count:'.format(len(dataList)))
    print(dataList[0])

    tokensFilePath = './data/{}_tokens.json'.format(mode)
    targetsFilePath = './data/{}_target.txt'.format(mode)
    if len(dataList) > 0:
        dataListJsonStr = json.dumps(dataList)
        with open(tokensFilePath, 'w') as tokensFile:
            tokensFile.write(dataListJsonStr)

        targetListJsonStr = json.dumps(targetList)
        with open(targetsFilePath, 'w') as targetsFile:
            targetsFile.write(targetListJsonStr)


def setupVobcabularyWithTokens(dataList):
    index2Word = list()
    word2Index = dict()
    lenList = list()
    for tokenList in dataList:
        lenList.append(len(tokenList))

        for token in tokenList:
            if token not in word2Index.keys():
                word2Index[token] = len(index2Word)
                index2Word.append(token)
    word2Index[UNKNOWN_TOKEN] = len(index2Word)
    index2Word.append(UNKNOWN_TOKEN)
    word2Index[PADDING_TOKEN] = len(index2Word)
    index2Word.append(PADDING_TOKEN)

    # print(len(index2Word))
    # print(len(word2Index))

    # x = range(len(lenList))
    # plt.bar(x, lenList)
    # plt.show()

    return word2Index, index2Word

def convertDataListToIndexList(dataList, word2Index):
    indexArray = np.zeros((len(dataList), MAX_TOKEN_LIST_SIZE), dtype=np.int32)
    for r, tokenList in enumerate(dataList):
        indexItemList = [word2Index.get(token, word2Index[UNKNOWN_TOKEN]) for token in tokenList]
        if len(indexItemList) < MAX_TOKEN_LIST_SIZE:
            indexItemList.extend([word2Index[PADDING_TOKEN]] * (MAX_TOKEN_LIST_SIZE - len(indexItemList)))
        elif len(indexItemList) > MAX_TOKEN_LIST_SIZE:
            indexItemList = indexItemList[:MAX_TOKEN_LIST_SIZE]

        for c, index in enumerate(indexItemList):
            indexArray[r, c] = index
    
    return indexArray

def getData(mode='train'):
    dataFilePath = './data/{}_tokens.json'.format(mode)
    targetFilePath = './data/{}_target.txt'.format(mode)
    dataList = list()
    targetList = list()
    with open(dataFilePath, 'r') as dataFile:
        dataList = json.loads(dataFile.read())

    with open(targetFilePath, 'r') as targetFile:
        targetList = json.loads(targetFile.read())

    word2Index, index2Word = setupVobcabularyWithTokens(dataList)  # 一般只要train数据的字词典
    X = convertDataListToIndexList(dataList, word2Index)
    y = np.array(targetList, dtype=np.int16)

    return X, y, len(word2Index), word2Index, index2Word


def getBatch(X, y, batchSize=BATCH_SIZE):
    assert X.shape[0] == y.shape[0], 'error shape: X = {}, y = {}'.format(X.shape, y.shape)

    shuffleIndexs = np.random.permutation(range(X.shape[0]))
    X = X[shuffleIndexs]
    y = y[shuffleIndexs]

    numBatches = int(X.shape[0] / BATCH_SIZE)
    for i in range(numBatches - 1):
        xBatch = X[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        yBatch = y[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        yield xBatch, yBatch



# if __name__ == '__main__':
    ## Warning: 重新读数据!!!!
    # readData('test')
    # readData('valid')
    # readData('train')

    # testList = [['11', '111'], ['a', 'aaaa'], ['b'], ['cc', 'ccc', 'cccc']]
    # jsonStr = json.dumps(testList)
    # print(jsonStr)
    # testList2 = json.loads(jsonStr)
    # print(testList2)

    # with open('./data/valid_tokens.json', 'r') as file:
    #     fileData = file.read()
    #     dataList = json.loads(fileData)
    #     print(len(dataList))
    #     setupVobcabularyWithTokens(dataList)

    # X, y = getData()
    # batch = getBatch(X, y)
    # xB, yB = next(batch)
    # print(xB.shape)
    # print(yB.shape)

    # arr1 = np.array([[1], [0], [1], [0]])
    # arr2 = arr1.squeeze()
    # print(arr2)

    # p = np.array([1, 2, 3, 1])
    # y = np.array([11, 22, 33, 11])
    # for pI, yI in zip(p, y):
    #     print('{}, {}'.format(pI, yI))