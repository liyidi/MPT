from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as scio
from tools import ops
import os
# ============================ 数据 ============================
datasetPath = '/home/liyd/myWork/dataset/AV163/seq11-1p-0100/'
'''load avdata(nor)'''
avdataPath = 'seq11-1p-0100_cam1_jpg/avdata_nor.mat'
avdataSavePath = os.path.join(datasetPath,avdataPath)
av_dataDict = scio.loadmat(avdataSavePath)
av_dataname = list(dict.keys(av_dataDict))[-1]
av_data = av_dataDict[av_dataname]
''' load audioWeight'''
audioWeightPath = 'seq11-1p-0100_cam1_jpg/s11c1_audioweight.mat'
audioWeightSavePath = os.path.join(datasetPath,audioWeightPath)
audioWeightDict = scio.loadmat(audioWeightSavePath)
audioWeightName = list(dict.keys(audioWeightDict))[-1]
audioWeight = audioWeightDict[audioWeightName]
'''load DataGT2D'''
DataGT2DPath = 'seq11-1p-0100_cam1_jpg/seq11-1p-0100_cam1_GT2D.mat'
DataGT2DSavePath = os.path.join(datasetPath,DataGT2DPath)
DataGT2DDict = scio.loadmat(DataGT2DSavePath)
DataGT2DName = list(dict.keys(DataGT2DDict))[-1]
startFr = 71
endFr = 549
DataGT2D = DataGT2DDict[DataGT2DName][startFr-1 : endFr]#标签格式为[x,y,w,h]，框框的左上角

'''generate visual weight'''
visualWeight = np.zeros(av_data.shape[0])
for dataFrame in range(av_data.shape[0]):
    vidataFr = av_data[dataFrame,9,:,:]
    GT2D =np.rint(DataGT2D[dataFrame,:]).astype(np.int)
    viInGT = vidataFr[GT2D[1]:GT2D[1]+GT2D[3],GT2D[0]:GT2D[0]+GT2D[2]]
    maxViGT = np.max(viInGT)
    visualWeight[dataFrame] = maxViGT
'''concatenate audioWeight and visualWeight, save as avWeightLabel'''
avdataWeight =  np.concatenate((audioWeight, visualWeight.reshape(-1,1)),axis = 1)
avWeightLabelPath = 'seq11-1p-0100_cam1_jpg/avWeightLabel.mat'
avWeightLabelSavePath = os.path.join(datasetPath,avWeightLabelPath)
#scio.savemat(avWeightLabelSavePath,dict([('avdataWeight',avdataWeight)]))
print('save as avWeightLabel.mat')
'''show the maps multiply avweight'''
# for dataFrame in range(av_data.shape[0]):
#     w = avdataWeight[dataFrame,:]
#     av = av_data[dataFrame,:,:,:]
#     x = av.reshape(av.shape[0], -1).T
#     output = np.matmul(x, w)
#     output = output.T
#     output = output.reshape(av.shape[1], -1)
#     ops.show_ColorMap(output,windowName = 'avdata')

