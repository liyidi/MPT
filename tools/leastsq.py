from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as scio
from tools import ops

#:generate the lstsqLabel from label,
# and save it as the same format as gaussLabel

# ============================ 数据 ============================
##load HDF5 object:av_data,av_dataName save the reference(index) of each frame's avdata
av_dataPath = '/home/yidi/dataset/avdataLists11m1.mat'
av_dataFile = h5py.File(av_dataPath)
av_dataName = av_dataFile[list(av_dataFile.keys())[1]][0]
'''load avdata_nor(from dataNormalize()),seq11-1p-0100_cam1_avdataNor.mat'''
avdataNorDict = scio.loadmat('/home/yidi/dataset/AV163/seq11-1p-0100/seq11-1p-0100_cam1_jpg/'
                             'seq11-1p-0100_cam1_avdataNor.mat')
avdataNor_name = list(dict.keys(avdataNorDict))[-1]
avdataNor = avdataNorDict[avdataNor_name]

##loadmat labelData, corresponding to av_data's size
label_dataDict = scio.loadmat('/home/yidi/dataset/AV163/seq11-1p-0100/seq11-1p-0100_cam1_jpg/'
                              'seq11-1p-0100_cam1_label_map.mat')
label_name = list(dict.keys(label_dataDict))[-1]
label_dataTotal = label_dataDict[label_name]
labelData = label_dataTotal[1:avdataNor.shape[0]+1]
#:x is avdata y is label, p is weight
##generate label
lstsqLabelNor_map = np.zeros((len(labelData),288,360))
lstsqLabelOrg_map = np.zeros((len(labelData),288,360))
for dataFrame in range(labelData.shape[0]):

    dataNor =avdataNor[dataFrame]
    label = labelData[dataFrame]
    xNor = dataNor.reshape(dataNor.shape[0], -1).T #:reshape the data and label to 1D vector
    yNor = label.reshape(1, -1).T
    pNor, resNor, rnkNor, sNor = lstsq(xNor, yNor)
    outputNor = np.matmul(xNor, pNor)
    outputNor = outputNor.T
    outputNor = outputNor.reshape(dataNor.shape[1], -1)
    lstsqLabelNor_map[dataFrame] = outputNor
    #Org
    dataOrg = np.array(av_dataFile[av_dataName[dataFrame]]).transpose((0,2,1))
    xOrg = dataOrg.reshape(dataOrg.shape[0], -1).T #:reshape the data and label to 1D vector
    yOrg = label.reshape(1, -1).T
    pOrg, resOrg, rnkOrg, sOrg = lstsq(xOrg, yOrg)
    outputOrg = np.matmul(xOrg, pOrg)
    outputOrg = outputOrg.T
    outputOrg = outputOrg.reshape(dataOrg.shape[1], -1)
    lstsqLabelOrg_map[dataFrame] = outputOrg

    # ops.show_ColorMap(outputNor,'winNor')
    # ops.show_ColorMap(outputOrg, 'winOrg')
    # ops.show_ColorMap(label,'winlabel')
    print('save frame:' + str(dataFrame + 1))
##save as seq11-1p-0100_cam1_lstsqLabel.mat
#scio.savemat('/home/yidi/dataset/AV163/seq11-1p-0100/seq11-1p-0100_cam1_jpg/seq11-1p-0100_cam1_lstsqLabel.mat',dict([('lstsqLabel_map',lstsqLabel_map)]))
scio.savemat('/home/yidi/dataset/AV163/seq11-1p-0100/seq11-1p-0100_cam1_jpg/'
             'seq11-1p-0100_cam1_lstsqLabelNor.mat',dict([('lstsqLabelNor_map',lstsqLabelNor_map)]))

# ops.show_ColorMap(output,'win1')
# ops.show_ColorMap(label,'win2')
##:print the location(index) of max value of output and label
# print(ops.argmax_ndarray(output))
# print(ops.argmax_ndarray(label))

print('---end---')