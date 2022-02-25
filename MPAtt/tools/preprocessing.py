'''load avdata and do preprocessing'''
import numpy as np
import os
import h5py
import scipy.io as scio
from tools import ops
from matplotlib import pyplot as plt

#:generate the lstsqLabel from label,
# and save it as the same format as gaussLabel

# ============================ 数据 ============================
datasetPath = '/home/liyd/myWork/dataset/AV163/seq11-1p-0100/'
avdataPath = 'seq11-1p-0100_cam1_jpg/avdata_org.mat'
avdataSavePath = os.path.join(datasetPath,avdataPath)
av_dataDict = scio.loadmat(avdataSavePath)
av_dataname = list(dict.keys(av_dataDict))[-1]
av_data = av_dataDict[av_dataname]

def dataNormalize(data, maxAudio_total ,maxVisual_total ):
    """
    :param data: the avData, size is (10,288,360)
    :return: the normalized avData, size is (10,288,360)
    normalize method: (x-min_local)/(max_total-min_local)
    """
    data_nor = np.zeros(data.shape)
    for i in range(9):
        audiodata = data[i]
        audiodata_nor = (audiodata - np.min(audiodata)) / (maxAudio_total - np.min(audiodata))
        data_nor[i] = audiodata_nor
    visualdata = data[9]
    visualdata_nor = (visualdata - np.min(visualdata)) / (maxVisual_total - np.min(visualdata))
    data_nor[9] = visualdata_nor
    return data_nor

maxAudio_total  = 10
maxVisual_total = 8
avdata_nor = np.zeros(list(av_data.shape))#initialize a zero container
for dataFrame in range(av_data.shape[0]):
    data = av_data[dataFrame]
    data_nor = dataNormalize(data, maxAudio_total ,maxVisual_total)
    avdata_nor[dataFrame] = data_nor
    # 打印信息
    if dataFrame % 50 == 0:
        print("normalize data of frame {:0>3}/{:0>3}".format(
            dataFrame, av_data.shape[0]))
'''save as avdata_nor.mat'''
avdataPath = 'seq11-1p-0100_cam1_jpg/avdata_nor.mat'
avdataSavePath = os.path.join(datasetPath,avdataPath)
scio.savemat(avdataSavePath,dict([('avdata_nor',avdata_nor)]))

'''see the max value of audio and visual data in every frames'''
maxAudio_curve = list()
maxVisual_curve = list()
for dataFrame in range(av_data.shape[0]):
    data = avdata_nor[dataFrame]
    maxAudio = np.max(data.reshape(10, -1)[0:9])
    #print('frame:{} maxAudio:{:.4f}'.format(dataFrame,maxAudio))
    maxAudio_curve.append(maxAudio)
    maxVisual = np.max(data[9])
    #print('frame:{} maxVisual:{:.4f}'.format(dataFrame,maxVisual))
    maxVisual_curve.append(maxVisual)
'''draw the max value'''
drawCurve_flag = True
if drawCurve_flag:
    a_x = range(len(maxAudio_curve))
    a_y = maxAudio_curve
    v_x = range(len(maxVisual_curve))
    v_y = maxVisual_curve
    plt.plot(a_x, a_y, label='maxAudio')
    plt.plot(v_x, v_y, label='maxVisual')

    plt.legend(loc='upper right')
    plt.ylabel('max value')
    plt.xlabel('frame')
    plt.show()
print('savemat as seq11-1p-0100_cam1_avdataNor.mat')
