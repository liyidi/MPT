from __future__ import absolute_import, division
import os
import scipy.io as scio
import torch.nn as nn
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import h5py

def loadh5pyFile(datasetPath, dataPath):
    """load h5pyfile and put into a ndarray container
    h5pyfile size is (*,360,288), we convert it into size(*,288,360)
    load HDF5 object:av_data,av_dataName save the reference(index) of each frame's avdata
    """
    au_dataPath = os.path.join(datasetPath, dataPath)
    au_dataFile = h5py.File(au_dataPath)
    au_dataName = au_dataFile[list(au_dataFile.keys())[1]][0]
    dim = au_dataFile[au_dataName[0]].shape
    data = np.zeros([au_dataName.shape[0],dim[0],dim[2],dim[1]])#initialize a zero container
    for dataFrame in range(au_dataName.shape[0]):
        au_datafr = np.array(au_dataFile[au_dataName[dataFrame]]).transpose((0,2,1))
        data[dataFrame] = au_datafr
    print('h5pyfile has been loaded')
    return data
def loadmat(datasetPath, dataPath):
    '''load .mat as ndarray'''
    dataFullPath = os.path.join(datasetPath,dataPath)
    dataDict = scio.loadmat(dataFullPath)
    name = list(dict.keys(dataDict))[-1]
    data = dataDict[name]
    return data

def splitDataset(data_dir,seqList,splitType,trainPct=0.9):
    if splitType == 'train&valid':
        trainList = list()
        validList = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                for seq in seqList:
                    if seq in sub_dir:
                        data_names = os.listdir(os.path.join(root, sub_dir))
                        data_names = list(filter(lambda x: x.endswith('.npz'), data_names))
                        # 遍历data
                        for i in range(len(data_names)):
                            if i < int(len(data_names)*trainPct):
                                data_name = data_names[i]
                                data_path = os.path.join(root, sub_dir, data_name)
                                trainList.append(data_path)
                            else:
                                data_name = data_names[i]
                                data_path = os.path.join(root, sub_dir, data_name)
                                validList.append(data_path)
    elif splitType == 'test':
        testList = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                for seq in seqList:
                    if seq in sub_dir:
                        data_names = os.listdir(os.path.join(root, sub_dir))
                        data_names = list(filter(lambda x: x.endswith('.npz'), data_names))
                        # 遍历data
                        for i in range(len(data_names)):
                            data_name = data_names[i]
                            data_path = os.path.join(root, sub_dir, data_name)
                            testList.append(data_path)
    elif splitType == 'fusetrain&valid':
        trainList = list()
        validList = list()
        for occdir in data_dir:
            occdata_dir = os.path.join(occdir, "sample")
            for root, dirs, _ in os.walk(occdata_dir):
                # 遍历类别
                for sub_dir in dirs:
                    for seq in seqList:
                        if seq in sub_dir:
                            data_names = os.listdir(os.path.join(root, sub_dir))
                            data_names = list(filter(lambda x: x.endswith('.npz'), data_names))
                            # 遍历data
                            for i in range(len(data_names)):
                                if i < int(len(data_names)*trainPct):
                                    data_name = data_names[i]
                                    data_path = os.path.join(root, sub_dir, data_name)
                                    trainList.append(data_path)
                                else:
                                    data_name = data_names[i]
                                    data_path = os.path.join(root, sub_dir, data_name)
                                    validList.append(data_path)
    if splitType == 'train&valid':
        return trainList,validList
    elif splitType == 'test':
        return testList
    elif splitType == 'fusetrain&valid':
        return trainList,validList

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


def show_image(img, boxes=None, frame_n = None, fig_n =1 ,box_fmt='ltwh',  colors=None,
               thickness=3,  delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, "frame:" + str(frame_n), (50, 200), font, 0.5, (255, 255, 255), 2)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)



    return

def show_response(response,frame, x_crop, r_sz, save_result):
    r_midsz = 8 * (r_sz - 1)  # 8*r_sz+56 - (8+56)
    response_large = cv2.resize(response, (r_midsz, r_midsz))  # the middle area
    response_pad = np.pad(response_large, ((64, 64), (64, 64)), 'constant', constant_values=(0, 0))
    response_pad = cv2.resize(response_pad, (x_crop.shape[0], x_crop.shape[0]))
    response_nor = cv2.normalize(response_pad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    heatmap = np.uint8(response_nor)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    frame_map = x_crop * 0.55 + heatmap * 0.45
    frame_map = frame_map.astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_map = cv2.putText(frame_map, "frame:" + str(frame), (50, 200), font, 0.5, (255, 255, 255), 2)
    cv2.imshow('response', frame_map)
    frame_map_name = str(1000+frame)[1:4] + '.jpg'
    if save_result:
        cv2.imwrite(os.path.join('/home/yidi/myprojects/siamese/my-siamfc-net/results/frame_map/',
                                 frame_map_name), frame_map)
    cv2.waitKey(1)
    return frame_map

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch_crop = img[corners[0]:corners[2], corners[1]:corners[3]]


    # resize to out_size
    patch = cv2.resize(patch_crop, (out_size, out_size),
                       interpolation=interp)

    return patch, patch_crop


def show_ColorMap(data,windowName): #:draw heatmap of ndarray data,
    # windowname is a userset name for imshow window
    data_nor = cv2.normalize(data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    heatmap = np.uint8(data_nor)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.namedWindow("heatmap: {} ".format(windowName), 0);
    cv2.resizeWindow("heatmap: {} ".format(windowName), 288, 360);
    cv2.imshow("heatmap: {} ".format(windowName), heatmap)#("show_ColorMap', heatmap)
    cv2.waitKey(1)
    return

def argmax_nograd(outputs): #:return the index of max value for tensor
    return torch.stack((outputs.view(outputs.size()[0], -1).argmax(1) // outputs.size()[-1],
                                       outputs.view(outputs.size()[0], -1).argmax(1) % outputs.size()[-1]), dim=0).t()


def argmax_ndarray(inputarray): #:return the index of max value for ndarray
    return np.where(inputarray == np.max(inputarray))


def soft_argmax2d(input,device,beta=100): #:argmax for tensor, have grad, need device
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(device)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result
