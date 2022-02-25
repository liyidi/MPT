"""
1.avdataCombine()--->avdata/avdata_org.mat
2.dataNormalize()--->avdata/avdata_nor.mat
3.getVisualWeight
4.getAudioWeight
5.concatenate audio-visual weight--->avdata/avweight.mat
6.save as separate samples
"""

import numpy as np
import os
import h5py
import scipy.io as scio
from tools import ops
from matplotlib import pyplot as plt
import glob
import cv2
# ============================ dataNormalize() ============================
def avdataNormalize(avdata_org,save_flag = False):
    maxAudio_total = np.max(avdata_org[:, 0:5, :, :])
    maxVisual_total = np.max(avdata_org[:, 5, :, :])
    avdata_nor = np.zeros(list(avdata_org.shape))  # initialize a zero container
    for dataFrame in range(avdata_org.shape[0]):
        data = avdata_org[dataFrame]
        data_nor = dataNormalize(data, maxAudio_total, maxVisual_total)
        avdata_nor[dataFrame] = data_nor
        # 打印信息
        if dataFrame % 50 == 0:
            print("normalize data of frame {:0>3}/{:0>3}".format(
                dataFrame, avdata_org.shape[0]))
    '''save as avdata_nor.mat'''
    if save_flag:
        avdataPath = 'avdata/' + sequence + '_cam' + str(cam_number) + '_avdata_nor.mat'
        avdataSavePath = os.path.join(datasetPath, avdataPath)
        scio.savemat(avdataSavePath, dict([('avdata_nor', avdata_nor)]))

    return avdata_nor
def dataNormalize(data, maxAudio_total ,maxVisual_total ):
    """
    :param data: the avData, size is (*,288,360)
    :return: the normalized avData, size is (*,288,360)
    normalize method: (x-min_local)/(max_total-min_local)
    """
    data_nor = np.zeros(data.shape)
    for i in range(6):
        audiodata = data[i]
        audiodata_nor = (audiodata - np.min(audiodata)) / (maxAudio_total - np.min(audiodata))
        data_nor[i] = audiodata_nor
    visualdata = data[5]
    visualdata_nor = (visualdata - np.min(visualdata)) / (maxVisual_total - np.min(visualdata))
    data_nor[5] = visualdata_nor
    return data_nor
# ============================ avdataCombine() ============================
def avdataCombine(sequence,cam_number,datasetPath,save_flag = False):
    '''audio data'''
    ##load HDF5 object:au_data,au_dataName save the reference(index) of each frame's avdata
    audioDataset = '../dataset/AAAI22_MPT/'
    audioPath = f'audio/gcfmaps_grid_{sequence}_cam{cam_number}mic3.mat'
    au_data = ops.loadh5pyFile(audioDataset, audioPath)
    '''visual data'''
    visualPath = f'visual/{sequence}_cam{cam_number}_heatmap.mat'
    vi_data = ops.loadmat(datasetPath, visualPath)
    '''combine audio and visual data'''
    # avdata_org = np.concatenate((au_data, vi_data.reshape(-1,1,288,360)), axis = 1)
    avdata_org = np.zeros([vi_data.shape[0],6,288,360])
    for dataFrame in range(vi_data.shape[0]):
        avdata_org[dataFrame] = np.concatenate((au_data[dataFrame], vi_data[dataFrame].reshape(1,288,360)), axis = 0)
        if dataFrame % 50 == 0:
            print("combine avdata of frame {:0>3}/{:0>3}".format(
                dataFrame, avdata_org.shape[0]))


    '''save as avdata_org.mat'''
    if save_flag:
        avdataPath = 'avdata/'+sequence+'_cam'+str(cam_number)+'_avdata_org.mat'
        avdataSavePath = os.path.join(datasetPath,avdataPath)
        scio.savemat(avdataSavePath,dict([('avdata_org',avdata_org)]))
        print("avdata_org has been saved")
    return avdata_org
'''generate visual weight'''
def getVisualWeight(avdata_nor,DataGT2D):
    visualWeight = np.zeros(avdata_nor.shape[0])
    for dataFrame in range(avdata_nor.shape[0]):
        vidataFr = avdata_nor[dataFrame,-1,:,:]
        GT2D =np.rint(DataGT2D[dataFrame,:]).astype(np.int)
        viInGT = vidataFr[GT2D[1]-1:GT2D[1]-1+GT2D[3],GT2D[0]-1:GT2D[0]-1+GT2D[2]]
        maxViGT = np.max(viInGT)
        visualWeight[dataFrame] = maxViGT
    return visualWeight
'''generate audio weight'''
def getAudioWeight(avdata_nor,audioDim,DataGT2D):
    audioDis = np.zeros([avdata_nor.shape[0],audioDim])
    for dataFrame in range(avdata_nor.shape[0]):
        audataFr = avdata_nor[dataFrame,0:5,:,:]
        loc = np.unravel_index(np.argmax(audataFr.reshape(5,-1),axis = 1), audataFr.shape)
        GT2D = DataGT2D[dataFrame,1]+DataGT2D[dataFrame,3]/2,DataGT2D[dataFrame,0]+DataGT2D[dataFrame,2]/2
        dis = np.sqrt(np.square(loc[1]-GT2D[0])+np.square(loc[2]-GT2D[1]))
        audioDis[dataFrame] = dis
    b, a, k = 0.35, np.mean(audioDis), 0.025
    sigmoidFc = lambda x: 1 - (2 * b / (np.exp(4 * k * (a - x)) + 1))
    audioWeight = np.zeros([avdata_nor.shape[0],audioDim])
    for dataFrame in range(avdata_nor.shape[0]):
        dis = audioDis[dataFrame]
        audioWeightFr = sigmoidFc(dis)
        audioWeight[dataFrame] = audioWeightFr
    print('get audioWeight')
    return audioWeight
'''concatenate audio-visual weight'''
def getavWeightSpv(audioWeight,visualWeight,avWeightPath,save_flag = False):
    avWeightSpv = np.concatenate((audioWeight,visualWeight.reshape(-1,1)),axis = 1)
    if save_flag:
        avWeightFile = f'{sequence}_cam{cam_number}_avweightSpv.mat'
        avWeightSavePath = os.path.join(avWeightPath, avWeightFile)
        scio.savemat(avWeightSavePath, dict([('avWeightSpv', avWeightSpv)]))
    print(f'save: {sequence}_cam{cam_number}_avweightSpv.mat ')
    return avWeightSpv
def getavWeightUnspv(avdata_nor,avWeightUnspvPath,save_flag = False):
    avWeightUnspv = np.zeros([avdata_nor.shape[0], 6])
    for dataFrame in range(avdata_nor.shape[0]):
        data = avdata_nor[dataFrame]
        label2 = np.zeros([6])  # label1: self-supervised, no gt
        loc = np.unravel_index(np.argmax(data.reshape(6, -1), axis=1), data.shape)
        for i in range(6):
            wp = data[:, loc[1][i], loc[2][i]]
            label2[i] = np.mean(wp[0:5]) + wp[-1]
        avWeightUnspv[dataFrame] = label2
    if save_flag:
        avWeightUnspvFile = f'{sequence}_cam{cam_number}_avweightUnspv.mat'
        avWeightUnspvSavePath = os.path.join(avWeightUnspvPath, avWeightUnspvFile)
        scio.savemat(avWeightUnspvSavePath, dict([('avWeightUnspv', avWeightUnspv)]))
    print(f'save: {sequence}_cam{cam_number}_avweightUnspv.mat ')
    return avWeightUnspv

#=============main=========================
def processing():
    '''1.avdataCombine: if avdata_org.mat exist, then load.'''
    avdataorgPath = os.path.join(datasetPath,'avdata')
    avdataorgFile = f'{sequence}_cam{cam_number}_avdata_org.mat'
    if os.path.exists(os.path.join(avdataorgPath,avdataorgFile)):
        avdata_org = ops.loadmat(avdataorgPath,avdataorgFile)
    else:
        if not os.path.exists(avdataorgPath):
            os.makedirs(avdataorgPath)
            print(f'step1: make folder: {avdataorgPath}')
        avdata_org = avdataCombine(sequence,cam_number,datasetPath,save_flag = False)
    print('step1: get original avdata: avdata_org')

    '''2.Normalize数据:if avdata_nor.mat exist, then load.'''
    avdatanorPath = os.path.join(datasetPath,'avdata')
    avdatanorFile = f'{sequence}_cam{cam_number}_avdata_nor.mat'
    if os.path.exists(os.path.join(avdatanorPath,avdatanorFile)):
        avdata_nor = ops.loadmat(avdatanorPath,avdatanorFile)
    else:
        if not os.path.exists(avdatanorPath):
            os.makedirs(avdatanorPath)
        avdata_nor = avdataNormalize(avdata_org, save_flag = False)
    print('step2: get normalized avdata: avdata_nor')

    '''load DataGT2D'''
    if sequence == 'seq11-1p-0100':
        startFRlist = [71, 70, 101]
        endFRlist = [549, 545, 578]
    elif sequence == 'seq08-1p-0100':
        startFRlist = [34, 28, 27]
        endFRlist = [515, 496, 513]
    elif sequence == 'seq12-1p-0100':
        startFRlist = [90, 124, 105]
        endFRlist = [1150, 1184, 1148]
    elif sequence == 'seq24-2p-0111':
        startFRlist = [0, 270, 0]
        endFRlist = [0, 520, 0]
    elif sequence == 'seq45-3p-1111':
        startFRlist = [0, 0, 230]
        endFRlist = [0, 0, 500]
    startFR, endFR = startFRlist[cam_number - 1], endFRlist[cam_number - 1]
    GTdatasetPath = '../dataset/AAAI22_MPT/'
    DataGT2DPath = f'seq/{sequence}_cam{cam_number}/GT/{sequence}_cam{cam_number}_GT2D.mat'
    DataGT2D = ops.loadmat(GTdatasetPath, DataGT2DPath)[startFR-1 : endFR]#标签格式为[img_x,img_y,w,h]，框框的左上角
    # ============================ generate weight============================

    '''1.supervised label(from gt): concatenate audio-visual weight'''
    avWeightPath = os.path.join(datasetPath,'label')
    avWeightFile = f'{sequence}_cam{cam_number}_avweightSpv.mat'
    if os.path.exists(os.path.join(avWeightPath,avWeightFile)):
        avWeightSpv = ops.loadmat(avWeightPath, avWeightFile)
    else:
        if not os.path.exists(avWeightPath):
            os.makedirs(avWeightPath)
        '''generate visual weight'''
        visualWeight = getVisualWeight(avdata_nor, DataGT2D)
        '''generate audio weight'''
        audioDim, visualDim = 5, 1
        audioWeight = getAudioWeight(avdata_nor, audioDim, DataGT2D)
        avWeightSpv = getavWeightSpv(audioWeight,visualWeight,avWeightPath,save_flag = False)
    print('step3: get labelSpv: avweightSpv(supervised)')

    '''2. Unsupervised label(no gt)'''
    avWeightUnspvPath = os.path.join(datasetPath,'label')
    avWeightUnspvFile = f'{sequence}_cam{cam_number}_avweightUnspv.mat'
    if os.path.exists(os.path.join(avWeightUnspvPath,avWeightUnspvFile)):
        avWeightUnspv = ops.loadmat(avWeightUnspvPath, avWeightUnspvFile)
    else:
        if not os.path.exists(avWeightUnspvPath):
            os.makedirs(avWeightUnspvPath)
        avWeightUnspv = getavWeightUnspv(avdata_nor,avWeightUnspvPath,save_flag = False)
    print('step3: get labelUnspv: avweightUnspv(Unsupervised)')

    # ============================ save as separate samples============================
    imgsetPath = '/home/liyd/myWork/dataset/AV163/'
    imgFilePath = f'{imgsetPath}{sequence}/{sequence}_cam{cam_number}_jpg/img'
    img_files = sorted(glob.glob(f'{imgFilePath}/*.jpg'))[startFRlist[cam_number - 1] - 1:endFRlist[cam_number - 1]]

    def showAttMap(data, img,dataFrame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, "frame:" + str(dataFrame), (50, 200), font, 0.5, (255, 255, 255), 2)
        data_nor = cv2.normalize(data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        heatmap = np.uint8(data_nor)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        f = heatmap * 0.5 + img * 0.5
        f = f.astype(np.uint8)
        cv2.imshow("attmap", f)  # ("show_ColorMap', heatmap)
        cv2.waitKey(1)

    for dataFrame in range(avdata_nor.shape[0]):
        folderPath =f'{datasetPath}/{sequence}_cam{cam_number}/'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        data = avdata_nor[dataFrame]
        labelSpv = avWeightSpv[dataFrame]
        labelUnspv = avWeightUnspv[dataFrame]
        imgPath = img_files[dataFrame]
        gt = DataGT2D[dataFrame,1]+DataGT2D[dataFrame,3]/2,DataGT2D[dataFrame,0]+DataGT2D[dataFrame,2]/2#(array_x,array_y) not img
        gtDiag = np.sqrt(np.sum(np.square(DataGT2D[dataFrame,2:4])))/2
        samplePath = f'{folderPath}{dataFrame:04}.npz'
        np.savez_compressed(samplePath, data=data,
                            labelSpv=labelSpv, labelUnspv = labelUnspv,
                            imgPath=imgPath,
                            seq=sequence, cam=cam_number, frame=dataFrame,
                            gt = gt, gtDiag = gtDiag)
        if dataFrame % 50 == 0:
            print("save data and label of frame {:0>3}/{:0>3}".format(
                dataFrame, avdata_org.shape[0]))
    print('end')


# ============================ start============================
sequence = '...'
trainType = '...'

if trainType == 'unOccTrain':
    datasetPath = '../dataset/AAAI22_MPT/sample/unOcc/'
elif trainType == 'inOccTrain':
    datasetPath = '../dataset/AAAI22_MPT/sample/inOcc/'

for cam_number in range(1,4):
    print(f'Now processing {sequence}_cam_number: {cam_number}')
    processing()