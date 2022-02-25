%%roomsetting画图估计一下房间范围。包括gt轨迹，三个相机（红色圈圈），坐标系原点（黄色），麦克风（绿色），以及数据库标定所用的标定点（蓝色，来自gt.m文件）。
load('E:\material_Learning\AV163\gt.mat');%load gt.mat

figure(51)
plot3(gt.cam_pos(1,:),gt.cam_pos(2,:),gt.cam_pos(3,:),'ro','markersize',5)%3个相机
hold on
plot3(0,0,0,'y.','markersize',25)%原点坐标
plot3(0,0.4,0,'go','markersize',5)%mic中心点坐标
hold on
grid on
plot3(gt.p3d(1,:),gt.p3d(2,:),gt.p3d(3,:),'b.','markersize',5)%