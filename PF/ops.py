from __future__ import absolute_import, division
import os
import torch.nn as nn
import scipy.io as scio
import cv2
import numpy as np
import torch

def loadmat(datasetPath, dataPath):
    '''load .mat as ndarray'''
    dataFullPath = os.path.join(datasetPath,dataPath)
    dataDict = scio.loadmat(dataFullPath)
    name = list(dict.keys(dataDict))[-1]
    data = dataDict[name]
    return data

def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def getSE(sequence,cam_number):#1-index
    if sequence == 'seq11-1p-0100':
        startFRlist = [71, 70, 101]
        endFRlist = [549, 545, 578]
    elif sequence == 'seq08-1p-0100':
        startFRlist = [34, 28, 27]
        endFRlist = [515, 496, 513]
    elif sequence == 'seq12-1p-0100':
        startFRlist = [90, 124, 105]
        endFRlist = [1150, 1184, 1148]
    startfr, endfr = startFRlist[cam_number-1], endFRlist[cam_number-1]
    return startfr, endfr

def argmax_ndarray(inputarray): #:return the index of max value for ndarray
    return np.where(inputarray == np.max(inputarray))