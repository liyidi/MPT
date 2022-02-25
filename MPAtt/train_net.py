# -*- coding: utf-8 -*-
"""
train the multimodal network
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from tools import ops
import sys
import cv2
from model.attNet import attNet
from model.mobileNet import MobileNetV3_Large
from tools.my_dataset import AVDataset
from tools.common_tools import set_seed
import random
import scipy.io as scio
from torch.autograd import Variable


spvType = 'unspv' #inSpv
OccOfTrain = 'unOccTrain'#unOccTrain inOccTrain
OccOfTest = 'inOccTest'#inOccTest
trainSeqList = ['seq01-1p-0100']#seq01,02,03
testSeqList = ['seq12-1p-0100']

'''set log path'''
date = f'0101'
log_dir = os.path.abspath(os.path.join(BASE_DIR, "log","MPAttModel_{0}.pth".format(date)))
'''set flag : train/test'''
train_flag       = True
saveNetwork_flag = True
drawCurve_flag   = True
test_flag        = False
os.environ["CUDA_VISIBLE_DEVICES"] = str()
saveNetOutput_flag  = False

datasetPath = '../dataset/AAAI22_MPT'

if OccOfTrain == 'inOccTrain':
    traindata_dir = os.path.join(datasetPath, "sample/inOcc")
    trainList, validList = ops.splitDataset(traindata_dir, trainSeqList, splitType='train&valid', trainPct=0.9)
elif OccOfTrain == 'unOccTrain':
    traindata_dir = os.path.join(datasetPath, "sample/unOcc")
    trainList,validList = ops.splitDataset(traindata_dir,trainSeqList,splitType='train&valid',trainPct = 0.9)

if OccOfTest == 'inOccTest':
    testdata_dir = os.path.join(datasetPath, "sample/inOcc")
elif OccOfTest == 'unOccTest':
    testdata_dir = os.path.join(datasetPath, "sample/unOcc")
testList = ops.splitDataset(testdata_dir,testSeqList, splitType='test')

# ============================ settings ============================
set_seed()  
MAX_EPOCH = 20
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1

train_data = AVDataset(spvType, dataList=trainList)
valid_data = AVDataset(spvType, dataList=validList)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
net = MobileNetV3_Large()
net = net.to(device)
criterionMSE = nn.MSELoss(reduction='mean')
lossFn = criterionMSE
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

if train_flag:
    train_curve = list()
    valid_curve = list()
    for epoch in range(MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels, _ , _ ,_= data
            inputs = Variable(torch.as_tensor(inputs, dtype=torch.float32).to(device), requires_grad = True)
            labels = Variable(torch.as_tensor(labels, dtype=torch.float32).to(device), requires_grad = True)
            outputsWeight = net(inputs)
            optimizer.zero_grad()
            loss = lossFn(outputsWeight, labels)
            loss.backward()
            optimizer.step()

            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean))
                loss_mean = 0.
        scheduler.step()

        if (epoch+1) % val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels, _ , _,_,_= data
                    inputs = torch.as_tensor(inputs, dtype=torch.float32).to(device)
                    labels = torch.as_tensor(labels, dtype=torch.float32).to(device)
                    outputsWeight = net(inputs)
                    loss = lossFn(outputsWeight, labels)
                    loss_val += loss.item()

                loss_val_epoch = loss_val / len(valid_loader)
                valid_curve.append(loss_val_epoch)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_epoch))
    if saveNetwork_flag:
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, log_dir)

if drawCurve_flag:
    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()

# ============================ inference ============================
if test_flag:
    checkpoint = torch.load(log_dir, map_location=device)
    net.load_state_dict(checkpoint['model'])
    test_data = AVDataset(spvType, dataList=testList)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    net.eval()
    disInPixelTotal = 0
    lossTotal = 0
    weightList = np.zeros([len(testList),6])
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # forward
            test_inputs, test_labels, imgPath, gt, frame = data
            test_inputs = torch.as_tensor(test_inputs, dtype=torch.float32).to(device)
            test_labels = torch.as_tensor(test_labels, dtype=torch.float32).to(device)
            outputs = net(test_inputs)
            test_loss = lossFn(outputs, test_labels)
            lossTotal += test_loss

            outputsArray = outputs.squeeze(0).detach().cpu().numpy()
            weightList[frame] = outputsArray

            Weight = outputs.unsqueeze(2).unsqueeze(3).expand_as(test_inputs)
            attmap = torch.mean(test_inputs * Weight, dim=1)
            test_outputsArray = attmap.squeeze(0).detach().cpu().numpy()
            locOutputs_xy = np.array([ops.argmax_ndarray(test_outputsArray)[0][0],
                                      ops.argmax_ndarray(test_outputsArray)[1][0]])

            gt = gt.detach().cpu().numpy()#max value's index of net output
            disInPixel = np.sqrt(np.sum(np.square(gt - locOutputs_xy)))# Euclidean Distance
            disInPixelTotal += disInPixel.item()
            imgFrame = ops.read_image(imgPath[0])
            print("sample:{:0>3} frame:{:0>4} Loss: {:.4f} disInPixel: {:.4f}"
                  .format(i, str(frame), test_loss, disInPixel))
    lossMean = lossTotal/len(test_loader)
    disMean = disInPixelTotal/len(test_loader)
    print("Total LossMean: {:.4f} disMean: {:.4f}".format(lossMean, disMean))
if saveNetOutput_flag:
    netOutputPath = f'net/{testSeqList[0]}_{spvType}_{OccOfTrain}_test{OccOfTest}.mat'
    netOutputSavePath = os.path.join(datasetPath,netOutputPath)
    scio.savemat(netOutputSavePath,dict([('weightList',weightList)]))
