from ParticleFilter import ParticleFilter
from ops import *
import glob
import matplotlib.pyplot as plt

def main():
    sequence, cam_number = 'seq12-1p-0100',1
    OccOfTest = 'unOcc'  # inOcc
    particles_num = 100
    startFR, endFR = getSE(sequence,cam_number)
    datasetPath = '/home/liyd/myWork/dataset/AAAI22_MPT/'
    DataGT2DPath = f'seq/{sequence}_cam{cam_number}/GT/{sequence}_cam{cam_number}_GT2D.mat'
    DataGT2D = loadmat(datasetPath, DataGT2DPath)[startFR-2: endFR]#[x,y,w,h]
    anno = DataGT2D[0]
    img_path = f'{datasetPath}seq/{sequence}_cam{cam_number}/img/'
    img_files = sorted(glob.glob(img_path + '*.jpg'))[startFR-2:endFR]
    netOutputPath = f'net/{sequence}_cam{cam_number}_{OccOfTest}Test.mat'
    netOutput = loadmat(datasetPath,netOutputPath)
    featurePath = os.path.join(datasetPath, f'sample/{OccOfTest}/{sequence}_cam{cam_number}/')
    out_path = os.path.join(datasetPath,f'PFoutput/{sequence}_cam{cam_number}_{OccOfTest}_PF')

    PF=ParticleFilter(anno,img_files,DataGT2D,netOutput,featurePath,out_path,particles_num,OccOfTest,saveFlag=False)
    while PF.img_index<len(PF.imgs)-1:
        PF.select()
        PF.propagate()
        PF.observe()
        result = PF.estimate()

    DataGT2D = loadmat(datasetPath, DataGT2DPath)[startFR-1: endFR]
    resave = np.zeros([DataGT2D.shape[0],7])
    disInPixelTotal = 0
    ACC = 0
    for dataFrame in range(DataGT2D.shape[0]):
        gt = DataGT2D[dataFrame, 0] + DataGT2D[dataFrame, 2] / 2, DataGT2D[dataFrame, 1] + DataGT2D[dataFrame, 3] / 2
        resave[dataFrame, 0:2] = result[dataFrame]
        resave[dataFrame,2:4] = gt
        disInPixel = np.sqrt(np.sum(np.square([int(gt[0]),int(gt[1])]- result[dataFrame])))
        resave[dataFrame, 4] = disInPixel
        disInPixelTotal += disInPixel.item()
        MAE = disInPixelTotal/(dataFrame+1)
        resave[dataFrame, 5] = MAE
        gtDiag = np.sqrt(np.sum(np.square(DataGT2D[dataFrame, 2:4])))
        if disInPixel > gtDiag * 0.5: ACC = ACC + 1
        resave[dataFrame, 6] = ACC
    plt.plot(resave[:,5],'r')
    plt.title('MAE for %s_cam%s = %.4f'%(sequence, cam_number, MAE))
    plt.xlabel("frame")
    plt.ylabel("MAE(pixel)")
    plt.grid()
    plt.show()
    print('end')

if __name__=='__main__':
    main()

