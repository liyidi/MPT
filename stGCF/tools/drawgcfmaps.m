function drawgcfmaps(frnum,data,obj,sample2d,sample3d,GT3d,cam_number,Data)
%data = gcfmap_v2{fr,1}
[X,Y] = meshgrid(1:obj.Width,1:obj.Height);
figure(1)
title(['cut = '])
for c = 1:size(data,2)
    subplot(3,3,c)    
    z = data(:,c);   
    B = griddata(sample2d(:,1),sample2d(:,2),z,X,Y);%,'v4'   
    surf(B)%imagesc()���ƶ�άͼ
    shading interp
    xlim([0 obj.Width])
    ylim([0 obj.Height])
    set(gca,'YDir','reverse');%y�ᷴ������ͼƬ����ϵһ�£�ʹԭ�������Ͻǡ�
    view([0,90])
    [maxVal maxInd] = max(data(:,c));    
    title(['fr = ' num2str(frnum) 'max =' num2str(maxVal)])
    hold on
    maxloc3d = [sample3d(maxInd,:,c) 1]';
    maxloc2d = project(maxloc3d,Data.cam,Data.align_mat) ;
    maxloc2d = maxloc2d(cam_number,:);%��һ���������
    plot3(maxloc2d(1),maxloc2d(2),maxVal,'m.','markersize',10)
if ~isempty(GT3d)
    gt3d = [GT3d 1]';
    gt2d = project(gt3d,Data.cam,Data.align_mat) ;
    gt2d = gt2d(cam_number,:);%��һ���������
    plot3(gt2d(1),gt2d(2),max(max(B)),'r.','markersize',10)
    hold on       
end       

%��ÿһ֡ͶӰ��ͼ��ƽ���gcfֵ��
end
pause(0.05)
clf()
end