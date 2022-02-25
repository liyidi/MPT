% This script defines initial parameters for the stgcfmap extracter.
%*************************************************************************%
%                        Initial Parameter
%*************************************************************************%
if strcmp(sequence, 'seq11-1p-0100')
    Para.audiolaglist = [6.64, 6.72, 5.52];
    Para.Blist = [1.5, 2.5, 2];
    Para.Ilist = [0.35, 0.2, 0.3];
    Para.startGTlist = [69,70,70];
    Para.endGTlist = [547,545,547];
elseif strcmp(sequence, 'seq08-1p-0100')
    Para.audiolaglist = [4.8, 5.6, 5.24];
    Para.Blist = [1.5, 2.5, 2.1];
    Para.Ilist = [0.35, 0.25, 0.3];
    Para.startGTlist = [14,28,18];
    Para.endGTlist = [495,496,504];
elseif strcmp(sequence, 'seq12-1p-0100')
    Para.audiolaglist = [0.44, -0.92, 0.52];
    Para.Blist = [1.5, 2.5, 2];
    Para.Ilist = [0.35, 0.2, 0.3];
    Para.startGTlist = [88,88,105];
    Para.endGTlist = [1148,1148,1148];
elseif strcmp(sequence, 'seq24-2p-0111')
    Para.audiolaglist = [21/25, -7/25, 1.6];
    Para.Blist = [1.5, 2.5, 2];
    Para.Ilist = [0.35, 0.25, 0.3];
    Para.startGTlist = [0,223,0];
    Para.endGTlist = [0,473,0];
elseif strcmp(sequence, 'seq45-3p-1111')
    Para.audiolaglist = [2+12/25, 18/25, 3+18/25];
    Para.Blist = [1.5, 2.5, 2];
    Para.Ilist = [0.35, 0.25, 0.3];
    Para.startGTlist = [0,0,187];
    Para.endGTlist = [0,0,457];
end
Para.map_num = 9;
Para.m1 = 14;
Para.m2 = 5;
Para.fs = 16000;
Para.c = 340; 
%*************************************************************************%
%                        Load Audio and Video
%*************************************************************************%
%1.load video
videofile=[sequence '_cam' num2str(cam_number) '_divx_audio.avi'];
Data.obj  =  VideoReader( [filepath sequence '\' videofile]);
%2.load audio sequence
Data.gt = importdata([filepath  '\gt.mat']);%load gt.mat
if Para.array_number == 1
    Data.mic_posgt = Data.gt.mic_pos(:,1:8)';
    for mic_number = 1:8
        audiofile=[sequence '_array1_mic' num2str(mic_number) '.wav'];
        [Data.audio{mic_number,1}, ] = audioread([filepath sequence '\' audiofile]);   
    end
elseif Para.array_number == 2
    Data.mic_posgt = Data.gt.mic_pos(:,9:16)';
    for mic_number = 1:8
    audiofile=[sequence '_array2_mic' num2str(mic_number) '.wav'];
    [Data.audio{mic_number,1}, ] = audioread([filepath sequence '\' audiofile]);
    end      
elseif Para.array_number == 3
    Data.mic_posgt = Data.gt.mic_pos';
    for mic_number = 1:8
    audiofile1=[sequence '_array1_mic' num2str(mic_number) '.wav'];%array1
    audiofile2=[sequence '_array2_mic' num2str(mic_number) '.wav'];%array2
    [Data.audio{mic_number,1}, ] = audioread([filepath sequence '\' audiofile1]);
    [Data.audio{mic_number+8,1}, ] = audioread([filepath sequence '\' audiofile2]);   
    end
end
%3.enframe the audio data
N=1/Data.obj.FrameRate*Para.fs;  
audiolag = Para.audiolaglist(cam_number);
if audiolag<0
    for mic_number = 1:size(Data.audio,1)   
        audiozero = [zeros(abs(audiolag*Para.fs),1); Data.audio{mic_number,1}];
        Data.frame{mic_number ,1}=enframe(audiozero,hamming(N),N/2); %frame_shift = N/2;
    end
else
    for mic_number = 1:size(Data.audio,1)
        Data.frame{mic_number ,1}=enframe(Data.audio{mic_number,1}(audiolag*Para.fs:end,1),hamming(N),N/2); %·ÖÖ¡¼Ó´°£¬frame_shift = N/2;
    end 
end

%1.load GT
Data.GT3D = importdata([filepath sequence '\' sequence '_myDataGT3D.mat']);
%2.load project file from AV16.3
Data.cam(1).Pmat = load([filepath '\CAM\camera1.Pmat.cal'], '-ASCII');
Data.cam(2).Pmat = load([filepath '\CAM\camera2.Pmat.cal'], '-ASCII');
Data.cam(3).Pmat = load([filepath '\CAM\camera3.Pmat.cal'], '-ASCII');
[ Data.cam(1).K, Data.cam(1).kc, Data.cam(1).alpha_c ] = readradfile([filepath '\CAM\cam1.rad']);
[ Data.cam(2).K, Data.cam(2).kc, Data.cam(2).alpha_c ] = readradfile([filepath '\CAM\cam2.rad']);
[ Data.cam(3).K, Data.cam(3).kc, Data.cam(3).alpha_c ] = readradfile([filepath '\CAM\cam3.rad']);
%3.Load certain matrix transformation data
load([filepath '\CAM\rigid010203.mat']);
Data.align_mat=rigid.Pmat;






