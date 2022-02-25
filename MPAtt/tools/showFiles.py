from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as scio
from tools import ops
import os
import glob
import cv2 as cv
def showAttMap(data,img,dataFrame):
    data_nor = cv.normalize(data, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    heatmap = np.uint8(data_nor)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    f = heatmap * 0.4 + img * 0.6
    f = f.astype(np.uint8)
    font = cv.FONT_HERSHEY_SIMPLEX
    # f = cv.putText(f, "frame:" + str(dataFrame), (50, 200), font, 0.5, (255, 255, 255), 2)
    cv.imshow("attmap", f)  # ("show_ColorMap', heatmap)
    cv.waitKey(1)

sequence,cam_number = 'seq11-1p-0100', 2
OccOfTest = 'inOcc'  # inOccTest
OccOfTrain = 'unOccTrain'
spvType = 'unspv'
if sequence == 'seq11-1p-0100':
    startFRlist = [71, 70, 101]
    endFRlist = [549, 545, 578]
elif sequence == 'seq08-1p-0100':
    startFRlist = [34, 28, 27]
    endFRlist = [515, 496, 513]
startFR, endFR = startFRlist[cam_number - 1], endFRlist[cam_number - 1]

GTdatasetPath = '/home/liyd/myWork/dataset/AV163/'
img_path = f'{GTdatasetPath}{sequence}/{sequence}_cam{cam_number}_jpg/img/'
img_files = sorted(glob.glob(img_path + '*.jpg'))[startFR - 1:endFR]

datasetPath = '/home/liyd/myWork/dataset/MMtracker/inOccTrain/goodScaler'#/goodScaler
samplePath = os.path.join(datasetPath, f'sample/{sequence}_cam{cam_number}/')

netOutdatasetPath = '/home/liyd/myWork/dataset/MMtracker/netOutput'
netOutputPath = f'{sequence}_cam{cam_number}_{spvType}_{OccOfTrain}_test{OccOfTest}Test.mat'
netOutput = ops.loadmat(netOutdatasetPath, netOutputPath)
im = '/home/liyd/myWork/dataset/MMtracker/0450.jpg'
#----show audio and visual feature
for dataFrame in range(70,480):
    img = cv.imread(img_files[dataFrame])
    sampleFile = f'{samplePath}{dataFrame:04}.npz'
    feature = np.load(sampleFile)['data']
    a = feature[0]
    showAttMap(a,img,dataFrame)
    v = feature[-1]
    showAttMap(v,img,dataFrame)
    #### cv.imwrite('/home/liyd/myWork/dataset/MMtracker/av3.jpg',f,[int(cv.IMWRITE_JPEG_QUALITY),100])
frameNum= len(os.listdir(samplePath))
#---show attention fusion map
for dataFrame in range(360,366):
    img = cv.imread(img_files[dataFrame])
    sampleFile = f'{samplePath}{dataFrame:04}.npz'
    feature = np.load(sampleFile)['data']
    # netOut = np.load(sampleFile)['labelUnspv']
    netOut = netOutput[dataFrame] #netOut is the weight of each feature
    fn = lambda i: netOut[i]*feature[i]
    attMapAu = sum([fn(i) for i in range(5)])
    attMap = fn(-1) + attMapAu/5
    showAttMap(attMap,img,dataFrame)


    # add the hanning window, center is self.state
    hannScale = 50
    window_influence = 0.15
    hann_window = np.outer(
        np.hanning(hannScale),
        np.hanning(hannScale))
    # ops.show_ColorMap(hann_window, 'hanning')

    hannMap = np.zeros([x + hannScale * 2 for x in attMap.shape])
    hannCenter = np.rint(np.load(sampleFile)['gt']) + [hannScale,hannScale]
    hannMap[int(hannCenter[0] - hannScale / 2):int(hannCenter[0] + hannScale / 2),
    int(hannCenter[1] - hannScale / 2):int(hannCenter[1] + hannScale / 2)] = hann_window
    hannMap = hannMap[hannScale:hannMap.shape[0] - hannScale, hannScale: hannMap.shape[1] - hannScale]
    attMapHann = (1 - window_influence) * attMap + \
               window_influence * hannMap
    showAttMap(attMapHann, img, dataFrame)
print('end')
#show the feature
for dataFrame in range(365,367):
    img = cv.imread(img_files[dataFrame])
    # cv.imshow("img", img)
    # cv.waitKey(1)
    sampleFile = f'{samplePath}{dataFrame:04}.npz'
    feature = np.load(sampleFile)['data']
    data = feature[0]
    data_nor = cv.normalize(data, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    heatmap = np.uint8(data_nor)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    f = heatmap.astype(np.uint8)
    cv.imshow("feature", f)  # ("show_ColorMap', heatmap)
    cv.waitKey(1)
print('end')