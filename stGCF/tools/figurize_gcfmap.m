figure(2)
plot3(Data.gt.cam_pos(1,:),Data.gt.cam_pos(2,:),Data.gt.cam_pos(3,:),'b.','markersize',15)
hold on
for i = 1:Para.map_num
plot3(sample3d(:,1,i),sample3d(:,2,i),sample3d(:,3,i),'m.','markersize',4)%3D
end
hold on
plot3(0,0,0,'y.','markersize',25)
hold on
plot3(Data.GT3D(startGT:endGT,2),Data.GT3D(startGT:endGT,3),Data.GT3D(startGT:endGT,4),'g.','markersize',8)%GT3D
grid on
%gt2d
figure(2)
for frmap = 1:size(gcfmap_max,1)
frgt = frmap+startGT-1;
FrameNumber = Data.GT3D(frgt,cam_number+4);  
imgframe = read(Data.obj,FrameNumber);
GT3d = Data.GT3D(frgt,2:4);
gt3d = [GT3d 1]';
gt2d = project(gt3d,Data.cam,Data.align_mat) ;
gt2d = gt2d(cam_number,1:2);
imshow(imgframe);
hold on
plot(gt2d(1),gt2d(2),'r.','markersize',5)
pause(0.01)
clf()
end
%gcfmap_max
[X,Y] = meshgrid(1:Data.obj.Width,1:Data.obj.Height);
for frmap = 1:size(gcfmap_max,1)
frgt = frmap+startGT-1;
FrameNumber = Data.GT3D(frgt,cam_number+4);  
GT3d = Data.GT3D(frgt,2:4);
data = gcfmap_max{frmap,1};
figure(1)
title(['cut = '])
for c = 1:size(data,2)
    subplot(1,5,c)    
    z = data(:,c);   
    B = griddata(sample2d(:,1),sample2d(:,2),z,X,Y);%,'v4'   
    surf(B)%imagesc()
    shading interp
    xlim([0 Data.obj.Width])
    ylim([0 Data.obj.Height])
    set(gca,'YDir','reverse');
    view([0,90])
    [maxVal maxInd] = max(data(:,c));    
    title(['fr = ' num2str(frmap) 'max =' num2str(maxVal/120)])
    hold on  
    %GT
    gt3d = [GT3d 1]';
    gt2d = project(gt3d,Data.cam,Data.align_mat) ;
    gt2d = gt2d(cam_number,:);
    plot3(gt2d(1),gt2d(2),max(max(B)),'r.','markersize',10)
    hold on
end       
pause(0.01)
clf()
end



