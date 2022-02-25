 # MPT
This is a Multi-modal Perception Tracker (MPT) for speaker tracking using both audio and visual modalities. 

We provide the *MATLAB&Python* implementation for our AAAI 2022 paper: **Multi-Modal Perception Attention Network with Self-Supervised Learning for Audio-Visual Speaker Tracking.**
## Requirements
![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg) ![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)  ![MATLAB ==2016](https://img.shields.io/badge/MATLAB-==2016-pink.svg)

## Data Preparation:
- **AV16.3:** the original dataset, available at [http://www.glat.info/ma/av16.3/](http://www.glat.info/ma/av16.3/)
- **MPTdata:** the preprocessed data provided for demo, available at.

## Descriptions:
1. **Audio Measurement:**  The MATLAB implement of stGCF. The parameter files that the camera projection model depends on can downloaded from [AV16.3](http://www.glat.info/ma/av16.3/) dataset.
2. **Visual Measurement:** A pre-trained Siamese network is employed to extract the response maps.
The PyTorch implementation of SiamFC tracker is described in the paper: [Fully-Convolutional Siamese Networks for Object Tracking](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).
3. **MPAtt Network:** The implement of proposed network.
*avdataCombine.py* is used firstly to integrate the audio and visual cues and normalize the data.
4. **PF:** The tracker is based on an improved PF algorithm.


## Citation
Please cite our paper if you find this repository useful in your resesarch:
```
@inproceedings{li2022mpt,
  Title= {Multi-Modal Perception Attention Network with Self-Supervised Learning for Audio-Visual Speaker Tracking},
  Author= {Yidi, Li and Hong, Liu and Hao, Tang},
  Booktitle= {AAAI},
  Year= {2022}
}
```


## Licence
This project is licensed under the terms of the MIT license.


