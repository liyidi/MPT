%%roomsetting��ͼ����һ�·��䷶Χ������gt�켣�������������ɫȦȦ��������ϵԭ�㣨��ɫ������˷磨��ɫ�����Լ����ݿ�궨���õı궨�㣨��ɫ������gt.m�ļ�����
load('E:\material_Learning\AV163\gt.mat');%load gt.mat

figure(51)
plot3(gt.cam_pos(1,:),gt.cam_pos(2,:),gt.cam_pos(3,:),'ro','markersize',5)%3�����
hold on
plot3(0,0,0,'y.','markersize',25)%ԭ������
plot3(0,0.4,0,'go','markersize',5)%mic���ĵ�����
hold on
grid on
plot3(gt.p3d(1,:),gt.p3d(2,:),gt.p3d(3,:),'b.','markersize',5)%