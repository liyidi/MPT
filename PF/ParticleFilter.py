import cv2 as cv
import glob
import numpy as np
from utils import *
import os
import math
from ops import *
import torch
def cap(x, lowerLimit, upperLimit):
    if x > upperLimit: x = upperLimit
    if x < lowerLimit: x = lowerLimit
    return x

def getHSV(img, particleState):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) / [180, 255, 255]
    y_lower = particleState.y - particleState.h_y
    y_upper = particleState.y + particleState.h_y
    x_lower = particleState.x - particleState.h_x
    x_upper = particleState.x + particleState.h_x
    y_lower = cap(y_lower,0,img_hsv.shape[0]-1)
    y_upper = cap(y_upper,0, img_hsv.shape[0]-1)
    x_lower = cap(x_lower, 0, img_hsv.shape[1]-1)
    x_upper = cap(x_upper, 0, img_hsv.shape[1]-1)
    subimage = img_hsv[y_lower:y_upper, x_lower: x_upper, :]
    bins = 8
    (rows, columns, _) = subimage.shape
    total = rows * columns
    cHist = np.zeros([bins, bins])
    imgBin = np.ceil(subimage * bins)
    imgBin[imgBin == 0] = 1

    for i in range(rows):
        for j in range(columns):
            hue = imgBin[i, j, 0]
            saturation = imgBin[i, j, 1]
            cHist[int(hue - 1), int(saturation - 1)] = cHist[int(hue - 1), int(saturation - 1)] + 1
    cHist = cHist / total
    return cHist

def getNetWeight(attMap, particleState):
    if particleState.y<0 or particleState.y>attMap.shape[0]-1\
        or particleState.x<0 or particleState.x>attMap.shape[1]-1:
        subattMap = 0
    else:
        subattMap = attMap[particleState.y,particleState.x]
    return subattMap

def showAttMap(data,img,out_path,img_index,saveFlag):
    data_nor = cv.normalize(data, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    heatmap = np.uint8(data_nor)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    f = heatmap * 0.5 + img * 0.5
    f = f.astype(np.uint8)
    cv.imshow("attmap", f)  # ("show_ColorMap', heatmap)
    cv.waitKey(1)
    if saveFlag:
        path = out_path + '/%04d(ATTMAP).jpg' % (img_index)
        cv.imwrite(path, f)

class state():
    def __init__(self,x,y,x_dot,y_dot,h_x,h_y,a_dot):
        self.x=x
        self.y=y
        self.x_dot=x_dot
        self.y_dot=y_dot
        self.h_x=h_x
        self.h_y=h_y
        self.a_dot=a_dot
    def draw_dot(self,img,out_path,img_index,saveFlag=False):
        cv.circle(img, center=(int(self.x), int(self.y)), radius=1, color=(0, 0, 255), thickness=1)
        cv.imshow('first', img)
        cv.waitKey(1)
        if saveFlag:
            path = out_path + '/0000ref.jpg'
            cv.imwrite(path, img)

    def draw_rectangle(self,a, img,out_path,img_index,saveFlag=False):
        cv.rectangle(img, (self.x - (a*self.h_x).astype(int),self.y - (a*self.h_y).astype(int)),
                     (self.x + (a*self.h_x).astype(int),self.y + (a*self.h_y).astype(int)), (0, 0, 255),thickness=2)
        self.img=img
        cv.imshow('first', img)
        cv.waitKey(1)
        if saveFlag:
            path = out_path + '/%04d.jpg' % (img_index)
            cv.imwrite(path, img)

class hist():
    def __init__(self,num=8,max_range=360.):  #HSV颜色空间，色调H（0-360度），饱和度S（0%-100%），明度V（0%-100%），在opencv中，H范围0-180，S范围0-255，V范围0-255
        self.num=num                          #直方图编号为0-7
        self.max_range=max_range
        self.divide=[max_range/num*i for i in range(num)]
        self.height=np.array([0. for i in range(num)])

    def get_hist_id(self,x):
        for i in range(self.num-1):
            if x>=self.divide[i] and x<self.divide[i+1]:
                return i
            elif x>=self.divide[-1] and x<=self.max_range:
                return self.num-1

    def update(self,i):
        self.height[i]+=1


class ParticleFilter():
    def __init__(self,anno,img_files,DataGT2D,netOutput,featurePath, out_path,particles_num,OccOfTest,saveFlag=False):
        self.saveFlag = saveFlag
        self.response = list()
        self.ACC = 0
        self.particles_num=particles_num
        self.out_path=out_path
        self.gt = DataGT2D
        self.netOutput = netOutput
        self.featurePath = featurePath
        self.DELTA_T=0.05
        self.VELOCITY_DISTURB=12.
        self.SCALE_DISTURB=0.0
        self.SCALE_CHANGE_D=0.0001
        self.img_index=0
        self.imgs = img_files
        self.errorTotal = 0
        self.OccOfTest = OccOfTest
        self.fristDiag = np.sqrt(np.sum(np.square(anno[2:4])))
        print(self.imgs[0])
        img_first = cv.imread(self.imgs[0])
        initial_state=state(x=int(anno[0]+anno[2]/2),y=int(anno[1]+anno[3]/2),x_dot=0.,y_dot=0.,h_x=int(anno[2]/2),h_y=int(anno[3]/2),a_dot=0.)
        self.state = initial_state
        self.particles=[]
        np.random.seed(8)
        random_nums=np.random.normal(0,0.4,(particles_num,7))
        self.weights = [1. / particles_num] * particles_num
        for i in range(particles_num):
            x0 = int(initial_state.x + random_nums.item(i, 0) * initial_state.h_x)
            y0 = int(initial_state.y + random_nums.item(i, 1) * initial_state.h_y)
            x_dot0 = initial_state.x_dot + random_nums.item(i, 2) * self.VELOCITY_DISTURB
            y_dot0 = initial_state.y_dot + random_nums.item(i, 3) * self.VELOCITY_DISTURB
            h_x0 = int(initial_state.h_x + random_nums.item(i, 4) * self.SCALE_DISTURB)
            h_y0 = int(initial_state.h_y + random_nums.item(i, 5) * self.SCALE_DISTURB)
            a_dot0 = initial_state.a_dot + random_nums.item(i, 6) * self.SCALE_CHANGE_D
            particle = state(x0, y0, x_dot0, y_dot0, h_x0, h_y0, a_dot0)
            particle.draw_dot(img_first,self.out_path,self.img_index,saveFlag=False)
            self.particles.append(particle)
        img_first = cv.imread(self.imgs[0])
        self.refHist = getHSV(img_first, initial_state)

    def select(self):
        self.img_index+=1
        self.img = cv.imread(self.imgs[self.img_index])
        index=get_random_index(self.weights)
        new_particles=[]
        for i in index:
            new_particles.append(state(self.particles[i].x,self.particles[i].y,self.particles[i].x_dot,self.particles[i].y_dot,self.particles[i].h_x,self.particles[i].h_y,self.particles[i].a_dot))
        dataFrame = self.img_index-1
        samplePath = f'{self.featurePath}{dataFrame:04}.npz'
        feature = np.load(samplePath)['data']
        netOut = self.netOutput[self.img_index-1]
        fn = lambda i: netOut[i]*feature[i]
        attMap = sum([fn(i) for i in range(6)])
        self.maxloc = argmax_ndarray(attMap)  #array_xy  not  particle xy
        for i in range(int(len(self.weights)*0.15)):################################################################33
            new_particles[i].x = int(self.maxloc[1])
            new_particles[i].y = int(self.maxloc[0])
        self.particles=new_particles

    def propagate(self):
        for particle in self.particles:
            random_nums = np.random.normal(0, 1, 7)
            particle.x = int(particle.x+particle.x_dot*self.DELTA_T+random_nums[0]*particle.h_x)
            particle.y = int(particle.y+particle.y_dot*self.DELTA_T+random_nums[1]*particle.h_y)
            particle.x_dot = 0+random_nums[2]*self.VELOCITY_DISTURB
            particle.y_dot = 0+random_nums[3]*self.VELOCITY_DISTURB
            particle.h_x = int(particle.h_x*(particle.a_dot+1)+random_nums[4]*self.SCALE_DISTURB+0.5)
            particle.h_y = int(particle.h_y*(particle.a_dot+1)+random_nums[5]*self.SCALE_DISTURB+0.5)
            particle.a_dot = particle.a_dot+random_nums[6]*self.SCALE_CHANGE_D

    def observe(self):
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        dataFrame = self.img_index-1
        samplePath = f'{self.featurePath}{dataFrame:04}.npz'
        feature = np.load(samplePath)['data']
        outputs = self.netOutput[dataFrame]
        if self.OccOfTest == 'inOcc':
            test_inputs = torch.as_tensor(feature, dtype=torch.float32).to(device)
            test_inputs = test_inputs.unsqueeze(0)
            outputs = torch.as_tensor(outputs, dtype=torch.float32).to(device)
            outputs = outputs.unsqueeze(0)
            Weight = outputs.unsqueeze(2).unsqueeze(3).expand_as(test_inputs)
            attmap = torch.mean(test_inputs * Weight, dim=1)
            attMap = attmap.squeeze(0).detach().cpu().numpy()
        else:
            netOut = self.netOutput[self.img_index-1]
            fn = lambda i: netOut[i]*feature[i]
            attMapau = sum([fn(i) for i in range(5)])
            attMap = fn(-1)+attMapau/5
        self.attMap = attMap
        hannScale = 50
        window_influence = 0.15
        hann_window = np.outer(np.hanning(hannScale),np.hanning(hannScale))
        hannMap = np.zeros([x+hannScale*2 for x in attMap.shape])
        hannCenter = [self.state.y+hannScale, self.state.x+hannScale]
        hannMap[int(hannCenter[0] - hannScale / 2):int(hannCenter[0] + hannScale / 2),
                int(hannCenter[1] - hannScale / 2):int(hannCenter[1] + hannScale / 2)] = hann_window
        hannMap = hannMap[hannScale:hannMap.shape[0] - hannScale, hannScale: hannMap.shape[1] - hannScale]
        self.attMapHann = (1 - window_influence) * attMap + \
                     window_influence * hannMap
        netweights = []
        for i in range(self.particles_num):
            w = getNetWeight(self.attMapHann, self.particles[i])
            netweights.append(w)
        weights = netweights
        b,k = 1,10
        a= np.max(netweights)
        y = (2 * b)/ (np.exp(4 * k * (a - weights)) + 1)#sigmoid function
        self.weights2 = y / sum(y)

    def estimate(self):
        self.weights = self.weights2
        self.state.x = np.sum(np.array([s.x for s in self.particles])*self.weights).astype(int)
        self.state.y = np.sum(np.array([s.y for s in self.particles])*self.weights).astype(int)
        self.state.h_x = np.sum(np.array([s.h_x for s in self.particles])*self.weights).astype(int)
        self.state.h_y = np.sum(np.array([s.h_y for s in self.particles])*self.weights).astype(int)
        self.state.x_dot = np.sum(np.array([s.x_dot for s in self.particles])*self.weights)
        self.state.y_dot = np.sum(np.array([s.y_dot for s in self.particles])*self.weights)
        self.state.a_dot = np.sum(np.array([s.a_dot for s in self.particles])*self.weights)

        gt = np.array([int(self.gt[self.img_index][0]+self.gt[self.img_index][2]/2),int(self.gt[self.img_index][1]+self.gt[self.img_index][3]/2)])
        loc = np.array([self.state.x,self.state.y])
        self.response.append(loc)
        gtDiag = np.sqrt(np.sum(np.square(self.gt[self.img_index, 2:4])))
        error = np.sqrt(np.sum(np.square(gt - loc)))  # Euclidean Distance
        self.errorTotal +=error
        errorMean = self.errorTotal/self.img_index
        if error> gtDiag*0.5: self.ACC = self.ACC+1
        if self.img_index % 50 == 0:
            print('frame: %s  error2d: %.2f errorMean: %.2f ACC: %s'  % (
                self.img_index, error, errorMean,self.ACC))

        if self.img_index == len(self.imgs)-1:
            return self.response
