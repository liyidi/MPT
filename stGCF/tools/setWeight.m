load('avdataLists11m1.mat')
obj  =  VideoReader( 'E:\material_Learning\AV163\seq11-1p-0100\seq11-1p-0100_cam1_divx_audio.avi');
%load project�������������������
Data.cam(1).Pmat = load('E:\material_Learning\code\TMM_AV_A_PF\Core_files\Data\camera1.Pmat.cal', '-ASCII' );
Data.cam(2).Pmat = load('E:\material_Learning\code\TMM_AV_A_PF\Core_files\Data\camera2.Pmat.cal', '-ASCII' );
Data.cam(3).Pmat = load('E:\material_Learning\code\TMM_AV_A_PF\Core_files\Data\camera3.Pmat.cal', '-ASCII' );
[ Data.cam(1).K, Data.cam(1).kc, Data.cam(1).alpha_c ] = readradfile( 'E:\material_Learning\code\TMM_AV_A_PF\Core_files\Data\cam1.rad' );
[ Data.cam(2).K, Data.cam(2).kc, Data.cam(2).alpha_c ] = readradfile( 'E:\material_Learning\code\TMM_AV_A_PF\Core_files\Data\cam2.rad' );
[ Data.cam(3).K, Data.cam(3).kc, Data.cam(3).alpha_c ] = readradfile( 'E:\material_Learning\code\TMM_AV_A_PF\Core_files\Data\cam3.rad' );
%Load certain matrix transformation data
load('E:\material_Learning\AV163\gt.mat');%load gt.mat
load('E:\material_Learning\code\TMM_AV_A_PF\Core_files\Data\rigid010203.mat');
Data.align_mat=rigid.Pmat;
clear rigid
%align_mat��cam����
align_mat = Data.align_mat;
cam = Data.cam;

%load GT����
sequence= char({ 'seq11-1p-0100'}); cam_number= 1;  % One-spekaer case
i = 1;
MouGT=fopen([sequence '-person' num2str(i) '-interpolated.3dmouthgt'],'r');
k=0;
while ~feof(MouGT)
buff = str2num(fgetl(MouGT)); %#ok<ST2NM>
if ~isempty( buff)
k = k+1;
DataMouthGT3D(k,:) = buff;
end
end
GT3D = DataMouthGT3D(69:547,:);
%����2D��
w = 9:18:obj.Width;
h = 9:18:obj.Height;
sample2d=[];%ÿ������2d�����������
for i = 1:size(w,2)
    for j = 1:size(h,2)
        sample2d = [sample2d;[w(i) h(j)]];
    end
end
%����sample2d����3d��
for i = 1:9
    sample3dcut = [];
    z = 0.5 + 0.5 * (i-1);
    for s = 1: size(sample2d,1)
    p2d = [sample2d(s,:) 1]';
    p3d = p2dtop3d(p2d,z,cam, align_mat);
    p3d= p3d(1:3,1)';
    sample3dcut = [sample3dcut;p3d];
    end
    sample3d(:,:,i) = sample3dcut;
end
% %9����ƵGCFmap����Ȩ�ؼ�Ȩ���
% for fr = 1:479
%     for z = 1:9
%         dis(z) = min(pdist2(sample3d(:,:,z),GT3D(fr,3:5),'Euclidean'));
%     end
%     w = 1./dis;
%     weight = (w - min(w))/(max(w)-min(w));
%     fh = @(x) avdataList{fr,1}(:,:,x).*weight(1,x);
%     R = arrayfun(fh,1:9,'UniformOutput',0);
%     summap = sum(cat(3,R{:}),3);
%     
%     figure(2)
%     subplot(2,1,1)
%     surf(summap)
%     shading interp
%     xlim([0 obj.Width])
%     ylim([0 obj.Height])
%     set(gca,'YDir','reverse');%y�ᷴ������ͼƬ����ϵһ�£�ʹԭ�������Ͻǡ�
%     view([0,90])
%     subplot(2,1,2)
%     imgframe = read(obj,fr+70);
%     imshow(imgframe);
%     title(['frame = ' num2str(fr) ])
%     hold on
%     pause(0.2)
% end
%9����ƵGCFmap+1����Ƶheatmap��Ȩ��ӣ���Ƶheatmap��Ȩ���Զ���
for fr = 1:479
    for z = 1:9
        dis(z) = min(pdist2(sample3d(:,:,z),GT3D(fr,3:5),'Euclidean'));
    end
    w = 1./dis;
    weight = (w - min(w))/(max(w)-min(w));
    weight = [weight, 2];
    fh = @(x) avdataList{fr,1}(:,:,x).*weight(1,x);
    R = arrayfun(fh,1:10,'UniformOutput',0);
    summap = sum(cat(3,R{:}),3);
    
    figure(2)
    subplot(2,1,1)
    surf(summap)
    shading interp
    xlim([0 obj.Width])
    ylim([0 obj.Height])
    set(gca,'YDir','reverse');%y�ᷴ������ͼƬ����ϵһ�£�ʹԭ�������Ͻǡ�
    view([0,90])
    subplot(2,1,2)
    imgframe = read(obj,fr+70);
    imshow(imgframe);
    title(['frame = ' num2str(fr) ])
    hold on
    pause(0.2)
end
%��ͼ��һ����Ƶheatmapÿһ֡�����ֵ��
for fr = 1:479
    maxlist(fr,1) = max(max(avdataList{fr,1}(:,:,10)));
    meanlist(fr,1)= mean(mean(avdataList{fr,1}(:,:,10)));
    stdlist(fr,1) =std2(avdataList{fr,1}(:,:,10));
    minlist(fr,1)= min(min(avdataList{fr,1}(:,:,10)));
end
figure(33)
plot(maxlist)
hold on
plot(meanlist)
hold on
plot(stdlist)
hold on
plot(minlist)
hold on
%loadģ��h=50 w=30(�Ŵ���)����Ƶheatmap
load('heatmap0607.mat')
for fr = 1:479
    maxlist2(fr,1) = max(max(heatmap(fr,:,:)));
    meanlist2(fr,1)= mean(mean(heatmap(fr,:,:)));
end
figure(3)
plot(maxlist2)
hold on
plot(meanlist2*100)
hold on

figure(4)
fr = 1
cla()
a = squeeze(heatmap(fr,:,:));
surf(a)
    shading interp
    xlim([0 obj.Width])
    ylim([0 obj.Height])
    set(gca,'YDir','reverse');%y�ᷴ������ͼƬ����ϵһ�£�ʹԭ�������Ͻǡ�
    view([0,90])
title(['frame = ' num2str(fr) 'max = ' num2str(maxlist2(fr,1))])
hold on
[maxVal maxInd] = max(max(a));%��fr֡����c����Ƭ��gcfֵ���ĵ�
[m,n]=find(a==maxVal);
plot3(n,m,maxVal,'b.','markersize',10)
fr = fr + 1
    

viweight = abs(maxlist - min(maxlist))/(7-4.5);

%9����ƵGCFmap+1����Ƶheatmap��Ȩ��ӣ���Ƶheatmap��Ȩ��Ϊviweight
for fr = 1:479
    for z = 1:9
        dis(z) = min(pdist2(sample3d(:,:,z),GT3D(fr,3:5),'Euclidean'));
    end
    w = 1./dis;
    weight = (w - min(w))/(max(w)-min(w));
    weight = [weight, viweight(fr)]
    fh = @(x) avdataList{fr,1}(:,:,x).*weight(1,x);
    R = arrayfun(fh,1:10,'UniformOutput',0);
    summap = sum(cat(3,R{:}),3);
    
    figure(2)
    subplot(2,1,1)
    surf(summap)
    shading interp
    xlim([0 obj.Width])
    ylim([0 obj.Height])
    set(gca,'YDir','reverse');%y�ᷴ������ͼƬ����ϵһ�£�ʹԭ�������Ͻǡ�
    view([0,90])
    subplot(2,1,2)
    imgframe = read(obj,fr+70);
    imshow(imgframe);
    title(['frame = ' num2str(fr) ])
    hold on
    pause(0.2)
end
%����10ά��Ȩ��
for fr = 1:479
    for z = 1:9
        dis(z) = min(pdist2(sample3d(:,:,z),GT3D(fr,3:5),'Euclidean'));
    end
    w = 1./dis;
    weight = (w - min(w))/(max(w)-min(w));
    weight = [weight, viweight(fr)];
    avweight(fr,:) = weight;
end
save('E:\material_Learning\code\GCF\avweights11m1.mat','avweight') 